#!/usr/bin/env python3
"""Quantize Atom01 policy.onnx (FP32) → INT8 weight .bin files + manifest.json.

Pipeline:
  1. Load policy.onnx via onnx, extract 4 weight tensors + 4 bias tensors.
  2. Symmetric per-tensor INT8 quantization: scale = max(|x|) / 127, q = round(x / scale).
  3. Write each quantized tensor as raw i8 bytes to weights/{w,b}{1..4}_int8.bin.
  4. Validate: dequantize, run inference, compare with FP32 reference (max relative error < 1%).
  5. Write weights/manifest.json with shapes, scales, validation result.
  6. Generate weights/test_fixtures.json with 10 random obs + expected outputs (Rust unit tests in Phase 2).

Usage:
  python3 tools/quantize_onnx.py \\
      --input  ../roboto_origin/modules/atom01_deploy/src/inference/models/policy.onnx \\
      --output weights/

If --input is missing or not a valid ONNX file, generates a synthetic policy for pipeline testing.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Final

import numpy as np
import onnx
import onnxruntime as ort

# Expected network topology (from policy.onnx architecture documented in SPEC §3.2)
EXPECTED_DIMS: Final[tuple[int, ...]] = (780, 512, 256, 128, 23)
EXPECTED_NUM_WEIGHTS: Final[int] = 4
EXPECTED_NUM_BIASES: Final[int] = 4
MAX_RELATIVE_ERROR: Final[float] = 0.01  # 1% threshold on significant outputs (matches SPEC §1.4 S3)
MIN_SIGNIFICANT_OUTPUT: Final[float] = 0.05  # outputs below this are treated as zero; relative error is meaningless there


@dataclass
class LayerArtifacts:
    name: str
    shape: list[int]
    weight_path: str
    bias_path: str
    weight_scale: float
    bias_scale: float
    weight_bytes: int
    bias_bytes: int


@dataclass
class Manifest:
    source_onnx: str
    input_shape: list[int]
    output_shape: list[int]
    layer_dims: list[int]
    layers: list[LayerArtifacts]
    validation_max_rel_err: float
    validation_passed: bool
    validation_samples: int


def symmetric_quantize(arr: np.ndarray) -> tuple[np.ndarray, float]:
    """Symmetric per-tensor INT8 quantization.

    Returns (q_array, scale) where:
        scale = max(|arr|) / 127
        q = round(arr / scale), clipped to [-128, 127]
    """
    max_abs = float(np.max(np.abs(arr)))
    if max_abs == 0.0:
        return np.zeros_like(arr, dtype=np.int8), 1.0
    scale = max_abs / 127.0
    q = np.clip(np.round(arr / scale), -128, 127).astype(np.int8)
    return q, scale


def dequantize(q: np.ndarray, scale: float) -> np.ndarray:
    """Reverse INT8 quantization back to FP32."""
    return q.astype(np.float32) * scale


def load_or_synthesize_policy(path: Path) -> onnx.ModelProto:
    """Load policy.onnx from path, or synthesize a structurally-valid one if missing."""
    if path.is_file():
        return onnx.load(str(path))
    print(f"[warn] {path} not found; synthesizing a structurally valid policy for pipeline testing",
          file=sys.stderr)
    return synthesize_policy()


def synthesize_policy() -> onnx.ModelProto:
    """Create a synthetic 5-layer MLP (input → 4 dense + elu) matching EXPECTED_DIMS."""
    rng = np.random.default_rng(seed=42)
    dims = EXPECTED_DIMS

    inputs = [onnx.helper.make_tensor_value_info("obs", onnx.TensorProto.FLOAT, [1, dims[0]])]
    outputs = [onnx.helper.make_tensor_value_info("action", onnx.TensorProto.FLOAT, [1, dims[-1]])]

    initializers: list[onnx.TensorProto] = []
    nodes: list[onnx.NodeProto] = []

    prev_name = "obs"
    for i in range(len(dims) - 1):
        in_d, out_d = dims[i], dims[i + 1]
        w = (rng.standard_normal((in_d, out_d)) * (1.0 / np.sqrt(in_d))).astype(np.float32)
        b = np.zeros((out_d,), dtype=np.float32)

        w_name = f"W{i}"
        b_name = f"b{i}"
        matmul_out = f"mm{i}"
        add_out = f"add{i}"

        initializers.append(onnx.numpy_helper.from_array(w, name=w_name))
        initializers.append(onnx.numpy_helper.from_array(b, name=b_name))

        nodes.append(onnx.helper.make_node("MatMul", [prev_name, w_name], [matmul_out]))
        nodes.append(onnx.helper.make_node("Add", [matmul_out, b_name], [add_out]))

        if i < len(dims) - 2:
            act_out = f"act{i}"
            nodes.append(onnx.helper.make_node("Elu", [add_out], [act_out]))
            prev_name = act_out
        else:
            prev_name = add_out

    nodes.append(onnx.helper.make_node("Identity", [prev_name], ["action"]))

    graph = onnx.helper.make_graph(nodes, "synthetic_policy", inputs, outputs, initializer=initializers)
    opset = onnx.helper.make_opsetid("", 17)
    return onnx.helper.make_model(graph, opset_imports=[opset], producer_name="atom01-quantize")


def extract_layers(model: onnx.ModelProto) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Extract 4 weight matrices and 4 bias vectors from the ONNX graph.

    Walks initializers in order; assumes MatMul→Add pairs W{i},b{i}.
    """
    initializers = sorted(model.graph.initializer, key=lambda t: t.name)
    weights: list[np.ndarray] = []
    biases: list[np.ndarray] = []
    for t in initializers:
        arr = onnx.numpy_helper.to_array(t).astype(np.float32)
        if "W" in t.name or "weight" in t.name.lower():
            weights.append(arr)
        elif "b" in t.name or "bias" in t.name.lower():
            biases.append(arr)
    if len(weights) != EXPECTED_NUM_WEIGHTS or len(biases) != EXPECTED_NUM_BIASES:
        raise ValueError(
            f"Expected {EXPECTED_NUM_WEIGHTS} weights and {EXPECTED_NUM_BIASES} biases, "
            f"got {len(weights)} and {len(biases)}"
        )
    return weights, biases


def write_int8_bin(path: Path, q: np.ndarray) -> int:
    """Write i8 array as raw bytes (row-major). Returns bytes written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(q.tobytes(order="C"))
    return q.nbytes


def validate(weights_q: list[np.ndarray], biases_q: list[np.ndarray],
             w_scales: list[float], b_scales: list[float],
             model: onnx.ModelProto) -> tuple[float, np.ndarray]:
    """Quantize-dequantize round-trip validation against FP32 reference.

    Returns (max_relative_error, fp32_outputs_for_test_fixtures).
    """
    sess = ort.InferenceSession(
        model.SerializeToString(),
        providers=["CPUExecutionProvider"],
    )
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name

    rng = np.random.default_rng(seed=123)
    samples = rng.standard_normal((10, EXPECTED_DIMS[0])).astype(np.float32)

    int8_outputs = []
    fp32_outputs = []
    for obs in samples:
        x = obs.reshape(1, -1)
        fp32_outputs.append(sess.run([output_name], {input_name: x})[0].flatten())

        h = x.T
        num_layers = len(weights_q)
        for i, (w_q, b_q, w_s, b_s) in enumerate(zip(weights_q, biases_q, w_scales, b_scales)):
            w = dequantize(w_q, w_s)
            b = dequantize(b_q, b_s)
            h = w.T @ h + b[:, None]
            if i < num_layers - 1:  # apply ELU only on hidden layers
                h = np.where(h >= 0, h, np.exp(h) - 1.0)
        int8_outputs.append(h.flatten())

    fp32_arr = np.array(fp32_outputs)
    int8_arr = np.array(int8_outputs)
    abs_err = np.abs(int8_arr - fp32_arr)
    significant = np.abs(fp32_arr) > MIN_SIGNIFICANT_OUTPUT
    if not significant.any():
        max_rel = 0.0
    else:
        rel_err = abs_err[significant] / np.abs(fp32_arr[significant])
        max_rel = float(np.max(rel_err))

    return max_rel, fp32_arr


def generate_test_fixtures(weights_q, biases_q, w_scales, b_scales, model, output_dir: Path) -> None:
    """Generate weights/test_fixtures.json with 10 random obs + FP32 reference outputs.

    Consumed by Phase 2 Rust unit tests in crates/mlp/tests/reference_test.rs.
    """
    sess = ort.InferenceSession(
        model.SerializeToString(),
        providers=["CPUExecutionProvider"],
    )
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name

    rng = np.random.default_rng(seed=456)
    samples = rng.standard_normal((10, EXPECTED_DIMS[0])).astype(np.float32)

    fixtures = {"samples": [], "scales": {
        "w": w_scales,
        "b": b_scales,
        "weight_bytes": [w.tobytes(order="C").hex() for w in weights_q],
        "bias_bytes": [b.tobytes(order="C").hex() for b in biases_q],
    }}
    for obs in samples:
        fp32_out = sess.run([output_name], {input_name: obs.reshape(1, -1)})[0].flatten()
        fixtures["samples"].append({
            "obs": obs.tolist(),
            "expected_action": fp32_out.tolist(),
        })

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "test_fixtures.json").write_text(json.dumps(fixtures, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(description="Quantize policy.onnx → INT8 .bin files")
    parser.add_argument("--input", type=Path,
                        default=Path("../roboto_origin/modules/atom01_deploy/src/inference/models/policy.onnx"),
                        help="Path to policy.onnx (FP32)")
    parser.add_argument("--output", type=Path, default=Path("weights"),
                        help="Output directory for .bin files and manifest")
    parser.add_argument("--strict", action="store_true",
                        help="Fail if max relative error exceeds 1%. Only meaningful with a "
                             "trained policy.onnx — random untrained weights give ~30%% error.")
    args = parser.parse_args()

    print(f"[info] Loading ONNX model: {args.input}")
    model = load_or_synthesize_policy(args.input)

    print("[info] Extracting weight/bias tensors...")
    weights, biases = extract_layers(model)

    weights_q: list[np.ndarray] = []
    biases_q: list[np.ndarray] = []
    w_scales: list[float] = []
    b_scales: list[float] = []
    layers: list[LayerArtifacts] = []

    for i, (w, b) in enumerate(zip(weights, biases), start=1):
        w_q, w_s = symmetric_quantize(w)
        b_q, b_s = symmetric_quantize(b)

        w_path = args.output / f"w{i}_int8.bin"
        b_path = args.output / f"b{i}_int8.bin"
        w_bytes = write_int8_bin(w_path, w_q)
        b_bytes = write_int8_bin(b_path, b_q)

        weights_q.append(w_q)
        biases_q.append(b_q)
        w_scales.append(w_s)
        b_scales.append(b_s)
        layers.append(LayerArtifacts(
            name=f"layer_{i}",
            shape=list(w.shape),
            weight_path=str(w_path.relative_to(args.output.parent)) if w_path.is_relative_to(args.output.parent) else str(w_path),
            bias_path=str(b_path.relative_to(args.output.parent)) if b_path.is_relative_to(args.output.parent) else str(b_path),
            weight_scale=w_s,
            bias_scale=b_s,
            weight_bytes=w_bytes,
            bias_bytes=b_bytes,
        ))
        print(f"[info] Layer {i}: W shape {w.shape} → {w_bytes} bytes, B shape {b.shape} → {b_bytes} bytes")

    print("[info] Validating INT8 vs FP32 reference...")
    max_rel, _ = validate(weights_q, biases_q, w_scales, b_scales, model)
    passed = max_rel < MAX_RELATIVE_ERROR
    print(f"[info] Max relative error: {max_rel:.6f} ({'PASS' if passed else 'FAIL'} < {MAX_RELATIVE_ERROR})")

    if not passed and args.strict:
        print(f"[error] Max relative error {max_rel} exceeds threshold {MAX_RELATIVE_ERROR}", file=sys.stderr)
        return 1

    manifest = Manifest(
        source_onnx=str(args.input),
        input_shape=[1, EXPECTED_DIMS[0]],
        output_shape=[1, EXPECTED_DIMS[-1]],
        layer_dims=list(EXPECTED_DIMS),
        layers=layers,
        validation_max_rel_err=max_rel,
        validation_passed=passed,
        validation_samples=10,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "manifest.json").write_text(json.dumps(asdict(manifest), indent=2))
    print(f"[info] Wrote manifest: {args.output / 'manifest.json'}")

    print("[info] Generating test fixtures for Phase 2 Rust unit tests...")
    generate_test_fixtures(weights_q, biases_q, w_scales, b_scales, model, args.output)
    print(f"[info] Wrote test fixtures: {args.output / 'test_fixtures.json'}")

    print(f"[done] Quantization complete. Outputs in {args.output}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
