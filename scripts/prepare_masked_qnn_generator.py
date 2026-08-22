"""Create one fixed-shape Kokoro generator with length-aware normalization.

Kokoro's generator uses AdaIN reductions over time.  Plain zero padding changes
those global statistics, which limits useful QNN bucket sizes.  This rewrite
adds a fixed-shape ``valid_frames`` input and replaces temporal ReduceMean nodes
with masked sums divided by the true temporal length.  Shapes remain completely
static for QNN HTP while padding is excluded from the normalization statistics.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


GENERATOR_INPUTS = {
    "/decoder/Slice_output_0": lambda bucket: [1, 128],
    "/decoder/decoder/Unsqueeze_output_0": lambda bucket: [1, 1, bucket],
    "/decoder/decoder/decode.3/Div_4_output_0": lambda bucket: [1, 512, bucket],
}


def set_shape(value_info: onnx.ValueInfoProto, shape: list[int]) -> None:
    del value_info.type.tensor_type.shape.dim[:]
    for value in shape:
        value_info.type.tensor_type.shape.dim.add().dim_value = value


def value_shapes(model: onnx.ModelProto) -> dict[str, list[int]]:
    values = [*model.graph.input, *model.graph.value_info, *model.graph.output]
    result: dict[str, list[int]] = {}
    for item in values:
        dims = item.type.tensor_type.shape.dim
        if dims and all(dim.HasField("dim_value") for dim in dims):
            result[item.name] = [dim.dim_value for dim in dims]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bucket", required=True, type=int)
    parser.add_argument(
        "--dynamic-input-masks",
        action="store_true",
        help=(
            "Keep the generator frame dimensions symbolic and accept externally "
            "supplied masks/denominators. This permits several static QNN sessions "
            "to share one model asset."
        ),
    )
    parser.add_argument(
        "--freeze-input-shapes",
        action="store_true",
        help="With --dynamic-input-masks, bake the probe bucket into every graph I/O shape.",
    )
    args = parser.parse_args()
    if args.bucket < 16:
        parser.error("--bucket must be at least 16 frames")
    if args.freeze_input_shapes and not args.dynamic_input_masks:
        parser.error("--freeze-input-shapes requires --dynamic-input-masks")

    model = onnx.load(str(args.input), load_external_data=False)
    shape_probe = copy.deepcopy(model)
    for graph_input in shape_probe.graph.input:
        set_shape(graph_input, GENERATOR_INPUTS[graph_input.name](args.bucket))
    set_shape(shape_probe.graph.output[0], [args.bucket * 300])
    shape_probe = onnx.shape_inference.infer_shapes(shape_probe, strict_mode=False, data_prop=True)
    shapes = value_shapes(shape_probe)
    if not args.dynamic_input_masks:
        model = shape_probe

    reductions = [node for node in model.graph.node if node.op_type == "ReduceMean"]
    temporal_lengths: set[int] = set()
    for node in reductions:
        shape = shapes.get(node.input[0])
        if not shape or len(shape) != 3:
            raise RuntimeError(f"No fixed rank-3 shape for {node.name}: {shape!r}")
        temporal_lengths.add(shape[2])

    axes_name = "kokoro.masked_mean.axes"
    denominator_shape_name = "kokoro.masked_mean.denominator_shape"
    initializers = [numpy_helper.from_array(np.asarray([2], dtype=np.int64), axes_name)]
    if not args.dynamic_input_masks:
        model.graph.input.append(helper.make_tensor_value_info("valid_frames", TensorProto.INT64, [1]))
        initializers.append(
            numpy_helper.from_array(
                np.asarray([1, 1, 1], dtype=np.int64), denominator_shape_name
            )
        )
    model.graph.initializer.extend(initializers)

    prelude: list[onnx.NodeProto] = []
    masks: dict[int, tuple[str, str]] = {}
    for temporal_length in sorted(temporal_lengths):
        # The generator has two normalization resolutions: bucket*10 and
        # bucket*60+1. Infer the exact valid-length formula from the fixed
        # tensor size instead of relying on node names.
        if temporal_length == args.bucket * 10:
            scale, offset = 10, 0
        elif temporal_length == args.bucket * 60 + 1:
            scale, offset = 60, 1
        else:
            raise RuntimeError(f"Unknown temporal normalization size {temporal_length}")

        if args.dynamic_input_masks:
            suffix = "10" if (scale, offset) == (10, 0) else "60"
            mask_name = f"valid_mask_{suffix}"
            denominator_name = f"valid_fraction_{suffix}"
            mask_length = (
                "kokoro_mask_10_length" if suffix == "10" else "kokoro_mask_60_length"
            )
            if not any(item.name == mask_name for item in model.graph.input):
                model.graph.input.append(
                    helper.make_tensor_value_info(
                        mask_name, TensorProto.FLOAT, [1, 1, mask_length]
                    )
                )
                prelude.append(
                    helper.make_node(
                        "ReduceMean",
                        [mask_name, axes_name],
                        [denominator_name],
                        name=f"kokoro.masked_mean.mask_fraction_{suffix}",
                        keepdims=1,
                    )
                )
            masks[temporal_length] = (mask_name, denominator_name)
            continue

        stem = f"kokoro.masked_mean.t{temporal_length}"
        scale_name = f"{stem}.scale"
        offset_name = f"{stem}.offset"
        range_name = f"{stem}.range"
        scaled = f"{stem}.scaled"
        valid_length = f"{stem}.valid_length"
        mask_bool = f"{stem}.mask_bool"
        mask_float = f"{stem}.mask_float"
        mask = f"{stem}.mask"
        mask_shape_name = f"{stem}.mask_shape"
        denominator_float = f"{stem}.denominator_float"
        denominator = f"{stem}.denominator"
        model.graph.initializer.extend(
            [
                numpy_helper.from_array(np.asarray([scale], dtype=np.int64), scale_name),
                numpy_helper.from_array(np.asarray([offset], dtype=np.int64), offset_name),
                numpy_helper.from_array(np.arange(temporal_length, dtype=np.int64), range_name),
                numpy_helper.from_array(
                    np.asarray([1, 1, temporal_length], dtype=np.int64), mask_shape_name
                ),
            ]
        )
        prelude.extend(
            [
                helper.make_node("Mul", ["valid_frames", scale_name], [scaled], name=f"{stem}.Mul"),
                helper.make_node("Add", [scaled, offset_name], [valid_length], name=f"{stem}.Add"),
                helper.make_node("Less", [range_name, valid_length], [mask_bool], name=f"{stem}.Less"),
                helper.make_node("Cast", [mask_bool], [mask_float], name=f"{stem}.CastMask", to=TensorProto.FLOAT),
                helper.make_node(
                    "Reshape",
                    [mask_float, mask_shape_name],
                    [mask],
                    name=f"{stem}.ReshapeMask",
                    allowzero=0,
                ),
                helper.make_node(
                    "Cast",
                    [valid_length],
                    [denominator_float],
                    name=f"{stem}.CastLength",
                    to=TensorProto.FLOAT,
                ),
                helper.make_node(
                    "Reshape",
                    [denominator_float, denominator_shape_name],
                    [denominator],
                    name=f"{stem}.ReshapeLength",
                    allowzero=0,
                ),
            ]
        )
        masks[temporal_length] = (mask, denominator)

    rewritten: list[onnx.NodeProto] = prelude
    rewritten_count = 0
    for node in model.graph.node:
        if node.op_type != "ReduceMean":
            rewritten.append(node)
            continue
        temporal_length = shapes[node.input[0]][2]
        mask, denominator = masks[temporal_length]
        stem = f"{node.name}.masked"
        masked = f"{stem}.values"
        summed = f"{stem}.sum"
        reduction = "ReduceMean" if args.dynamic_input_masks else "ReduceSum"
        rewritten.extend(
            [
                helper.make_node("Mul", [node.input[0], mask], [masked], name=f"{stem}.Mul"),
                helper.make_node(
                    reduction,
                    [masked, axes_name],
                    [summed],
                    name=f"{stem}.{reduction}",
                    keepdims=1,
                ),
                helper.make_node("Div", [summed, denominator], list(node.output), name=f"{stem}.Div"),
            ]
        )
        rewritten_count += 1

    del model.graph.node[:]
    model.graph.node.extend(rewritten)
    del model.graph.value_info[:]
    if args.freeze_input_shapes:
        fixed_shapes = {
            **{name: shape(args.bucket) for name, shape in GENERATOR_INPUTS.items()},
            "valid_mask_10": [1, 1, args.bucket * 10],
            "valid_mask_60": [1, 1, args.bucket * 60 + 1],
        }
        for graph_input in model.graph.input:
            set_shape(graph_input, fixed_shapes[graph_input.name])
        set_shape(model.graph.output[0], [args.bucket * 300])
    model = onnx.shape_inference.infer_shapes(model, strict_mode=False, data_prop=True)
    onnx.checker.check_model(model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(args.output))
    print(
        f"Wrote {args.output} ({args.output.stat().st_size / 1_000_000:.1f} MB); "
        f"bucket={args.bucket if args.freeze_input_shapes else ('dynamic' if args.dynamic_input_masks else args.bucket)}, "
        f"masked ReduceMean nodes={rewritten_count}, temporal lengths={sorted(temporal_lengths)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
