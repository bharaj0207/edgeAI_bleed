#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export PyTorch/TorchScript model to ONNX")
    parser.add_argument("--input", required=True, help="Path to .pt/.pth model")
    parser.add_argument("--output", required=True, help="Path to output ONNX")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--input-shape", required=True, help="Comma-separated shape e.g. 1,3,224,224")
    parser.add_argument("--input-name", default="input")
    parser.add_argument("--output-name", default="output")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("torch is required for export_pytorch.py") from exc

    shape = [int(v.strip()) for v in args.input_shape.split(",") if v.strip()]
    if not shape:
        raise SystemExit("--input-shape must include at least one dimension")

    model_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = None
    try:
        model = torch.jit.load(str(model_path), map_location="cpu")
    except RuntimeError:
        model = torch.load(str(model_path), map_location="cpu")

    if hasattr(model, "eval"):
        model.eval()
    else:
        raise SystemExit("Loaded object is not a PyTorch module. Provide a scripted/traced/module file.")

    dummy = torch.randn(*shape)

    torch.onnx.export(
        model,
        dummy,
        str(output_path),
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=[args.input_name],
        output_names=[args.output_name],
        dynamic_axes=None,
    )

    print(f"Exported ONNX model to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
