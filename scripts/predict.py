"""Run DA-SFNet inference on an image, directory, video, or webcam."""

import argparse
from pathlib import Path

from _bootstrap import ROOT  # noqa: F401
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Image/directory/video path or webcam index")
    parser.add_argument("--weights", type=Path, default=ROOT / "weights/da-sfnet-best.pt")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--device", default="")
    parser.add_argument("--project", type=Path, default=ROOT / "runs/predict")
    parser.add_argument("--name", default="da_sfnet")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.weights.is_file():
        raise FileNotFoundError(args.weights)
    source = int(args.source) if args.source.isdigit() else args.source
    YOLO(str(args.weights)).predict(
        source=source,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        save=True,
        project=str(args.project),
        name=args.name,
        exist_ok=True,
    )


if __name__ == "__main__":
    main()

