"""Evaluate the released DA-SFNet checkpoint on FloorPlanCAD."""

import argparse
import multiprocessing as mp
import os
from pathlib import Path

from _bootstrap import ROOT  # noqa: F401
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=ROOT / "weights/da-sfnet-best.pt")
    parser.add_argument("--data", type=Path, default=ROOT / "data/FloorPlanCAD/data.yaml")
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--project", type=Path, default=ROOT / "runs/val")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.chdir(ROOT)
    for required in (args.weights, args.data):
        if not required.is_file():
            raise FileNotFoundError(required)

    metrics = YOLO(str(args.weights)).val(
        data=str(args.data),
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        iou=0.7,
        max_det=300,
        plots=True,
        project=str(args.project),
        name=f"da_sfnet_{args.split}",
        exist_ok=True,
    )
    print(f"Precision: {metrics.box.mp:.6f}")
    print(f"Recall:    {metrics.box.mr:.6f}")
    print(f"mAP50:     {metrics.box.map50:.6f}")
    print(f"mAP50-95:  {metrics.box.map:.6f}")


if __name__ == "__main__":
    mp.freeze_support()
    main()
