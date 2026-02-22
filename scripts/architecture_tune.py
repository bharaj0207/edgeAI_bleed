#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Architecture tuning placeholder hook")
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--input-shape", required=True)
    parser.add_argument("--notes", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "input_shape": args.input_shape,
        "notes": args.notes,
        "status": "no-op",
        "guidance": [
            "Integrate a pruning/distillation script and export a new ONNX into this iteration directory.",
            "Update pipeline config to point model path to the tuned artifact for the next iteration.",
        ],
    }
    (work_dir / "architecture_tuning.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
