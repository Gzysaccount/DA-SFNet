"""Train DA-SFNet on FloorPlanCAD with the configuration used in the paper."""

import argparse
import multiprocessing as mp
import os
from pathlib import Path

from _bootstrap import ROOT  # noqa: F401
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / "configs/models/da-sfnet.yaml")
    parser.add_argument("--data", type=Path, default=ROOT / "data/FloorPlanCAD/data.yaml")
    parser.add_argument("--weights", type=Path, default=ROOT / "weights/yolo11n.pt")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0", help="CUDA device, e.g. 0, or cpu")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--project", type=Path, default=ROOT / "runs/train")
    parser.add_argument("--name", default="da_sfnet_seed1_m30")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.chdir(ROOT)
    for required in (args.model, args.data):
        if not required.is_file():
            raise FileNotFoundError(required)

    if args.resume:
        resume_ckpt = ROOT / "runs/train" / args.name / "weights/last.pt"
        if not resume_ckpt.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_ckpt}")
        YOLO(str(resume_ckpt)).train(resume=True)
        return

    model = YOLO(str(args.model))
    if args.weights.is_file():
        model.load(str(args.weights))
    else:
        print(f"Warning: pretrained weights not found at {args.weights}; training from scratch.")

    model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        cache=False,
        device=args.device,
        workers=args.workers,
        optimizer="SGD",
        lr0=0.01,
        momentum=0.937,
        warmup_epochs=3.0,
        amp=True,
        seed=args.seed,
        deterministic=True,
        close_mosaic=30,
        pretrained=True,
        project=str(args.project),
        name=args.name,
        exist_ok=False,
    )


if __name__ == "__main__":
    mp.freeze_support()
    main()
