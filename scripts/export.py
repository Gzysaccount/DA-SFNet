"""Export the released DA-SFNet checkpoint to a deployment format."""

import argparse
from pathlib import Path

from _bootstrap import ROOT  # noqa: F401
from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=ROOT / "weights/da-sfnet-best.pt")
    parser.add_argument("--format", default="onnx", help="onnx, engine, openvino, torchscript, etc.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if not args.weights.is_file():
        raise FileNotFoundError(args.weights)
    path = YOLO(str(args.weights)).export(format=args.format, imgsz=args.imgsz, device=args.device)
    print(path)


if __name__ == "__main__":
    main()

