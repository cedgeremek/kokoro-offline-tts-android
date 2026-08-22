#!/usr/bin/env python3
"""Build a fixed, length-masked Kokoro v1 acoustic graph for QNN.

The v1 generator contains temporal AdaIN ReduceMean operations.  A plain
zero-padded static bucket changes those statistics and produces audible
artifacts.  This tool replaces each temporal ReduceMean with a masked sum
divided by the real (unpadded) length, then exposes a small FP32 I/O contract
around the FP16 graph for Android/ORT.

This is deliberately a build-time tool.  It writes only to the path supplied
by the caller and never modifies installed application assets.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper, shape_inference


def _dims(value_info):
    return [d.dim_value or d.dim_param for d in value_info.type.tensor_type.shape.dim]


def _set_dims(value_info, dims):
    shape = value_info.type.tensor_type.shape
    del shape.dim[:]
    for d in dims:
        dim = shape.dim.add()
        if isinstance(d, int):
            dim.dim_value = d
        else:
            dim.dim_param = str(d)


def _find_value(graph, name):
    for collection in (graph.input, graph.value_info, graph.output):
        for value in collection:
            if value.name == name:
                return value
    raise KeyError(name)


def _make_static(model, bucket):
    g = model.graph
    acoustic_output = "/decoder/decoder/generator/conv_post/Conv_output_0"
    # The three cut tensors are the exact v1 acoustic contract.
    _set_dims(_find_value(g, "/decoder/Slice_output_0"), [1, 128])
    _set_dims(_find_value(g, "/decoder/decoder/Unsqueeze_output_0"), [1, 1, bucket])
    _set_dims(_find_value(g, "/decoder/decoder/decode.3/Div_4_output_0"), [1, 512, bucket])
    _set_dims(_find_value(g, acoustic_output), [1, 22, 60 * bucket + 1])
    inferred = shape_inference.infer_shapes(model)
    # Preserve the graph-input/output shape edits after inference.
    g = inferred.graph
    _set_dims(_find_value(g, "/decoder/Slice_output_0"), [1, 128])
    _set_dims(_find_value(g, "/decoder/decoder/Unsqueeze_output_0"), [1, 1, bucket])
    _set_dims(_find_value(g, "/decoder/decoder/decode.3/Div_4_output_0"), [1, 512, bucket])
    acoustic_value = _find_value(g, acoustic_output)
    _set_dims(acoustic_value, [1, 22, 60 * bucket + 1])

    # The source generator also contains iSTFT and waveform post-processing.
    # Merely changing graph.output leaves those nodes live and can cause the
    # QNN compiler to bind the waveform shape to an output named "acoustic".
    # Retain only the backward slice that produces conv_post.
    required = {acoustic_output}
    retained = []
    for node in reversed(g.node):
        if any(name in required for name in node.output):
            retained.append(node)
            required.update(name for name in node.input if name)
    retained.reverse()
    del g.node[:]
    g.node.extend(retained)
    del g.output[:]
    g.output.append(acoustic_value)
    return inferred


def _add_initializer(graph, name, array):
    if not any(t.name == name for t in graph.initializer):
        graph.initializer.append(numpy_helper.from_array(np.asarray(array), name=name))


def _wrap_fp32_io(model, bucket, output_name, mask_casts, fp32_internal=False):
    g = model.graph
    old_inputs = list(g.input[:3])
    old_output = g.output[0]
    existing_nodes = list(g.node)
    original_output_name = old_output.name
    old_input_names = [x.name for x in old_inputs]
    # The original v1 QNN candidate is FP16 internally.  Keep the app-facing
    # contract FP32 so Java can continue using FloatBuffer/FloatArray and
    # avoid Float16 JNI plumbing.  For the larger buckets we also support a
    # numerically safer FP32-internal graph: the HTP compiler may still lower
    # safe subgraphs to FP16, but the exported graph never performs an
    # unconditional lossy FP32->FP16 conversion at its boundary.
    internal_names = []
    casts = []
    if fp32_internal:
        internal_names = old_input_names
        internal_output = output_name
        # Source v1 FP32 inputs/output are already the Android contract.  Do
        # not insert casts; this is important because FP16 conversion of the
        # acoustic graph overflows on ordinary sentence lengths.
        old_output.name = internal_output
        old_output.type.tensor_type.elem_type = TensorProto.FLOAT
        for node in existing_nodes:
            for i, name in enumerate(node.input):
                if name in old_input_names:
                    node.input[i] = name
            for i, name in enumerate(node.output):
                if name == original_output_name:
                    node.output[i] = internal_output
    else:
        for value in old_inputs:
            value.type.tensor_type.elem_type = TensorProto.FLOAT
            internal = value.name + "__fp16"
            internal_names.append(internal)
            cast = helper.make_node("Cast", [value.name], [internal], name=internal + "_cast", to=TensorProto.FLOAT16)
            casts.append(cast)
        for node in g.node:
            for i, name in enumerate(node.input):
                if name in old_input_names:
                    node.input[i] = internal_names[old_input_names.index(name)]
        internal_output = output_name + "__fp16"
        old_output.name = internal_output
        old_output.type.tensor_type.elem_type = TensorProto.FLOAT16
        for node in existing_nodes:
            for i, name in enumerate(node.output):
                if name == original_output_name:
                    node.output[i] = internal_output
    output = helper.make_tensor_value_info(output_name, TensorProto.FLOAT, [1, 22, 60 * bucket + 1])
    del g.node[:]
    # All casts must be before the rewritten graph, and the output cast last.
    g.node.extend(casts)
    g.node.extend(mask_casts)
    g.node.extend(existing_nodes)
    if not fp32_internal:
        g.node.append(helper.make_node("Cast", [old_output.name], [output_name], name=output_name + "_cast", to=TensorProto.FLOAT))
    del g.output[:]
    g.output.append(output)
    return model


def build(source, output, bucket, mean_mask=False, fp32_internal=False):
    model = onnx.load(str(source))
    if not model.graph.input or not model.graph.output:
        raise ValueError("source graph has no IO")
    model = _make_static(model, bucket)
    g = model.graph
    data_by_name = {v.name: v for v in list(g.value_info) + list(g.input) + list(g.output)}

    # App-facing FP32 mask/length inputs.  The FP16 graph casts these masks;
    # the FP32-safe graph keeps them FP32 throughout.
    mask10 = helper.make_tensor_value_info("valid_mask_10", TensorProto.FLOAT, [1, 1, 10 * bucket])
    mask60 = helper.make_tensor_value_info("valid_mask_60", TensorProto.FLOAT, [1, 1, 60 * bucket + 1])
    len10 = helper.make_tensor_value_info("valid_length_10", TensorProto.FLOAT, [1])
    len60 = helper.make_tensor_value_info("valid_length_60", TensorProto.FLOAT, [1])
    g.input.extend([mask10, mask60] if mean_mask else [mask10, mask60, len10, len60])

    casts = []
    cast_names = ("valid_mask_10", "valid_mask_60") if mean_mask else (
        "valid_mask_10", "valid_mask_60", "valid_length_10", "valid_length_60"
    )
    mask_suffix = "__fp32" if fp32_internal else "__fp16"
    for name in cast_names:
        internal = name + mask_suffix
        if fp32_internal:
            # Use the public name directly; aliases keep the replacement
            # logic independent of the source graph's names.
            casts.append(helper.make_node("Identity", [name], [internal], name=internal + "_identity"))
        else:
            casts.append(helper.make_node("Cast", [name], [internal], name=internal + "_cast", to=TensorProto.FLOAT16))
    if mean_mask:
        casts.extend([
            helper.make_node("ReduceMean", [f"valid_mask_10{mask_suffix}", "/encoder/predictor/text_encoder/lstms.1/Constant_output_0"], [f"valid_fraction_10{mask_suffix}"], name="valid_fraction_10", keepdims=1),
            helper.make_node("ReduceMean", [f"valid_mask_60{mask_suffix}", "/encoder/predictor/text_encoder/lstms.1/Constant_output_0"], [f"valid_fraction_60{mask_suffix}"], name="valid_fraction_60", keepdims=1),
        ])

    # The exporter leaves intermediate value shapes symbolic, so use the
    # stable v1 generator stages to classify the two temporal resolutions:
    # noise_res is 10*B and resblocks is 60*B+1.
    replacements = []
    rewritten = 0
    for node in g.node:
        if node.op_type != "ReduceMean" or len(node.input) < 2:
            replacements.append(node)
            continue
        if "/noise_res." in node.name:
            match = re.search(r"/noise_res\.(\d+)/", node.name)
            if match is not None and int(match.group(1)) == 0:
                mask, denom = f"valid_mask_10{mask_suffix}", (f"valid_fraction_10{mask_suffix}" if mean_mask else f"valid_length_10{mask_suffix}")
            else:
                mask, denom = f"valid_mask_60{mask_suffix}", (f"valid_fraction_60{mask_suffix}" if mean_mask else f"valid_length_60{mask_suffix}")
        elif "/resblocks." in node.name:
            match = re.search(r"/resblocks\.(\d+)/", node.name)
            if match is not None and int(match.group(1)) <= 2:
                mask, denom = f"valid_mask_10{mask_suffix}", (f"valid_fraction_10{mask_suffix}" if mean_mask else f"valid_length_10{mask_suffix}")
            else:
                mask, denom = f"valid_mask_60{mask_suffix}", (f"valid_fraction_60{mask_suffix}" if mean_mask else f"valid_length_60{mask_suffix}")
        else:
            replacements.append(node)
            continue
        # Preserve the original axes input (opset 20 ReduceMean uses it as a
        # second input) and keepdims semantics exactly.
        stem = node.name or node.output[0]
        masked = stem + "__masked"
        summed = stem + "__sum"
        divided = node.output[0]
        reduction = "ReduceMean" if mean_mask else "ReduceSum"
        replacements.extend([
            helper.make_node("Mul", [node.input[0], mask], [masked], name=stem + "_mask_mul"),
            helper.make_node(reduction, [masked, node.input[1]], [summed], name=stem + "_sum", keepdims=1),
            helper.make_node("Div", [summed, denom], [divided], name=stem + "_divide"),
        ])
        rewritten += 1
    if rewritten == 0:
        raise RuntimeError(f"no temporal ReduceMean nodes matched bucket {bucket}")
    # Keep the original nodes plus the new mask/denom casts in an internal
    # side list; _wrap_fp32_io will rebuild the final node ordering.
    # Install the replacement graph in-place.  The external mask/length
    # inputs are already FP32 and are converted by the casts below.
    del g.node[:]
    g.node.extend(replacements)
    g.node.extend([])
    output_name = "acoustic"
    model = _wrap_fp32_io(model, bucket, output_name, casts, fp32_internal=fp32_internal)
    model.ir_version = min(model.ir_version, 10)
    onnx.checker.check_model(model)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(output))
    return rewritten


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--bucket", required=True, type=int)
    ap.add_argument("--mean-mask", action="store_true", help="use masked ReduceMean plus mask fraction")
    ap.add_argument("--fp32-internal", action="store_true", help="preserve FP32 arithmetic instead of casting the source graph to FP16")
    args = ap.parse_args()
    count = build(args.source, args.output, args.bucket, args.mean_mask, args.fp32_internal)
    print(f"wrote {args.output} bucket={args.bucket} rewritten_reduce_mean={count} fp32_internal={args.fp32_internal}")


if __name__ == "__main__":
    main()
