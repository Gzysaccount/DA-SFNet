"""Train the four configurations in the DA-SFNet two-module ablation study."""

import argparse
import gc
import multiprocessing as mp
import os
from pathlib import Path

from _bootstrap import ROOT  # noqa: F401
import torch
from ultralytics import YOLO


VARIANTS = {
    "baseline": ROOT / "configs/models/baseline.yaml",
    "mst_lada": ROOT / "configs/models/mst-lada.yaml",
    "da_fft": ROOT / "configs/models/da-fft.yaml",
    "da_sfnet": ROOT / "configs/models/da-sfnet.yaml",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--data", type=Path, default=ROOT / "data/FloorPlanCAD/data.yaml")
    parser.add_argument("--weights", type=Path, default=ROOT / "weights/yolo11n.pt")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--project", type=Path, default=ROOT / "runs/ablation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.chdir(ROOT)
    for required in (args.data, args.weights):
        if not required.is_file():
            raise FileNotFoundError(required)

    for variant in args.variants:
        model = YOLO(str(VARIANTS[variant])).load(str(args.weights))
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
            name=f"{variant}_seed{args.seed}_m30",
            exist_ok=False,
        )
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    mp.freeze_support()
    main()

