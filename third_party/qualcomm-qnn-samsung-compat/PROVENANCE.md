# Qualcomm QNN 2.4 Samsung device-probe compatibility build

The local AAR is derived mechanically from the official Maven artifact
`com.qualcomm.qti:onnxruntime-android-qnn:2.4.0` (SHA-256
`48F0AD8ACD0864D4DBA66DB283468C1D082E9D1B91B33D92F3CE40562E0C533D`).

`scripts/build_samsung_qnn_2_4_compat.ps1` changes the provider's single
12-byte FastRPC device-name probe from `fastrpc-cdsp` to Samsung's
`adsprpc-smd` (NUL-padded). No model execution, provider option, or QNN runtime
code is modified. This is a bounded diagnostic/compatibility build for the
Galaxy S24 Ultra SM-S928/SM8650. Qualcomm's upstream 2.5.0 source replaces this
device-node heuristic with Android SoC-manufacturer detection.

Upstream source: https://github.com/onnxruntime/onnxruntime-qnn/tree/v2.4.0

The upstream repository declares the provider source under the MIT license.
