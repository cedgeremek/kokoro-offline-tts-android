#!/usr/bin/env python3
"""Build a v1.5.1 QNN-compatible CMUdict frontend APK.

The v1.5.1 bytecode expects two indexed ``*.mlex`` files.  This tool emits
that same compact index format from the exact CMU IPA source that was bundled
by the v1.4.2 pronunciation build, substitutes only those two payloads in the
already-qualified v1.5.1 APK, advances the manifest version, zip-aligns, and
signs with the local Android debug key.  It deliberately does *not* modify
model, voices, QNN contexts, native libraries, or runtime bytecode.
"""

from __future__ import annotations

import argparse
import hashlib
import struct
import subprocess
import tempfile
import zipfile
from pathlib import Path


VERSION_NAME_OLD = b"1.5.1-qnn-b256-b384-misaki-s24"
VERSION_NAME_NEW = b"1.5.2-qnn-b256-b384-pronun-s24"
VERSION_CODE_NEW = 19
VERSION_CODE_RESOURCE_ID = 0x0101021B
MLEX_NAMES = ("assets/misaki_en_us.mlex", "assets/misaki_en_gb.mlex")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_cmudict(source: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    for line in source.read_text(encoding="utf-8").splitlines():
        if not line or "\t" not in line:
            continue
        word, phones = line.split("\t", 1)
        # v1.4.2 looked up the same CMU entry case-insensitively.  The MLEX
        # reader has a lower-case fallback, so normalized ASCII keys preserve
        # that behavior while maintaining binary-search ordering.
        key = word.lower()
        if all(ord(character) < 128 for character in key) and key not in entries:
            entries[key] = phones.strip()

    # These are the v1.4.2 deterministic pronunciation additions.  They are
    # not Misaki data and prevent the old letter-name fallback for the reported
    # OOV words while retaining every normal CMU pronunciation unchanged.
    entries.update(
        {
            "cutie": "kjˈuTi",
            "markedness": "mˈɑɹkədnəs",
        },
    )
    if len(entries) < 120_000:
        raise ValueError(f"CMU dictionary unexpectedly small: {len(entries)} entries")
    return entries


def build_mlex(entries: dict[str, str]) -> bytes:
    rows = [f"{word}\t{phones}\n".encode("utf-8") for word, phones in sorted(entries.items())]
    offsets: list[int] = []
    position = 0
    for row in rows:
        offsets.append(position)
        position += len(row)
    return b"MLEX" + struct.pack("<I", len(rows)) + b"".join(
        struct.pack("<I", offset) for offset in offsets
    ) + b"".join(rows)


def patch_manifest(manifest: bytes) -> bytes:
    data = bytearray(manifest)
    old_name_utf16 = VERSION_NAME_OLD.decode("ascii").encode("utf-16le")
    new_name_utf16 = VERSION_NAME_NEW.decode("ascii").encode("utf-16le")
    if old_name_utf16 not in data:
        raise ValueError("Expected v1.5.1 version-name string was not found in AndroidManifest")
    if len(old_name_utf16) != len(new_name_utf16):
        raise AssertionError("Version-name replacement must be in-place")
    name_offset = data.index(old_name_utf16)
    data[name_offset : name_offset + len(old_name_utf16)] = new_name_utf16

    resource_ids: list[int] = []
    version_code_updates = 0
    # The outer RES_XML_TYPE is one container chunk.  Its payload begins at
    # byte 8 and contains the string pool/resource map/start-element chunks.
    offset = 8
    while offset + 8 <= len(data):
        chunk_type, header_size, chunk_size = struct.unpack_from("<HHI", data, offset)
        if chunk_size < header_size or offset + chunk_size > len(data):
            raise ValueError(f"Malformed binary XML chunk at {offset}")
        if chunk_type == 0x0180:  # RES_XML_RESOURCE_MAP_TYPE
            resource_ids = list(struct.unpack_from(f"<{(chunk_size - header_size) // 4}I", data, offset + header_size))
        elif chunk_type == 0x0102:  # RES_XML_START_ELEMENT_TYPE
            attribute_extension = offset + 16  # ResXMLTree_node header is always 16 bytes.
            attribute_start, attribute_size, attribute_count = struct.unpack_from(
                "<HHH", data, attribute_extension + 8
            )
            for index in range(attribute_count):
                attribute = attribute_extension + attribute_start + index * attribute_size
                name_index = struct.unpack_from("<I", data, attribute + 4)[0]
                resource_id = resource_ids[name_index] if name_index < len(resource_ids) else 0
                if resource_id == VERSION_CODE_RESOURCE_ID:
                    # Res_value starts at +12.  Android uses TYPE_INT_DEC here;
                    # change only the typed payload, never the string pool.
                    struct.pack_into("<I", data, attribute + 16, VERSION_CODE_NEW)
                    version_code_updates += 1
        offset += chunk_size
    if version_code_updates != 1:
        raise ValueError(f"Expected exactly one versionCode attribute, found {version_code_updates}")
    return bytes(data)


def clone_info(info: zipfile.ZipInfo) -> zipfile.ZipInfo:
    clone = zipfile.ZipInfo(info.filename, info.date_time)
    clone.comment = info.comment
    clone.extra = info.extra
    clone.internal_attr = info.internal_attr
    clone.external_attr = info.external_attr
    clone.create_system = info.create_system
    clone.create_version = info.create_version
    clone.extract_version = info.extract_version
    clone.flag_bits = info.flag_bits
    clone.compress_type = info.compress_type
    return clone


def is_legacy_signature(name: str) -> bool:
    upper = name.upper()
    return upper == "META-INF/MANIFEST.MF" or (
        upper.startswith("META-INF/") and upper.endswith((".SF", ".RSA", ".DSA", ".EC"))
    )


def repack(input_apk: Path, output_apk: Path, mlex: bytes) -> None:
    with zipfile.ZipFile(input_apk, "r") as source, zipfile.ZipFile(output_apk, "w", allowZip64=True) as target:
        replacements = {name: mlex for name in MLEX_NAMES}
        seen = set()
        for info in source.infolist():
            if is_legacy_signature(info.filename):
                continue
            payload = source.read(info.filename)
            if info.filename == "AndroidManifest.xml":
                payload = patch_manifest(payload)
            elif info.filename in replacements:
                payload = replacements[info.filename]
                seen.add(info.filename)
            target.writestr(clone_info(info), payload)
        if seen != set(MLEX_NAMES):
            raise ValueError(f"Missing expected frontend assets: {set(MLEX_NAMES) - seen}")


def run(*command: str) -> None:
    print("+", " ".join(command))
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-apk", type=Path, required=True)
    parser.add_argument("--cmudict", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--build-tools", type=Path, required=True)
    parser.add_argument("--keystore", type=Path, required=True)
    args = parser.parse_args()

    entries = read_cmudict(args.cmudict)
    mlex = build_mlex(entries)
    print(f"CMU frontend: {len(entries)} entries, {len(mlex)} bytes, sha256={hashlib.sha256(mlex).hexdigest()}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="kokoro-v151-cmu-") as directory:
        root = Path(directory)
        unsigned = root / "unsigned.apk"
        aligned = root / "aligned.apk"
        repack(args.input_apk, unsigned, mlex)
        run(str(args.build_tools / "zipalign.exe"), "-f", "-P", "16", "4", str(unsigned), str(aligned))
        run(
            str(args.build_tools / "apksigner.bat"), "sign",
            "--ks", str(args.keystore),
            "--ks-key-alias", "androiddebugkey",
            "--ks-pass", "pass:android",
            "--key-pass", "pass:android",
            "--v1-signing-enabled", "true",
            "--v2-signing-enabled", "true",
            "--out", str(args.output),
            str(aligned),
        )
    print(f"Wrote {args.output} ({args.output.stat().st_size} bytes, sha256={sha256(args.output)})")


if __name__ == "__main__":
    main()
