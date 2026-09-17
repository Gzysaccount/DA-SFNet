"""Build DA-SFNet and execute one dummy forward pass."""

from _bootstrap import ROOT  # noqa: F401
import torch

from ultralytics import YOLO


def main() -> None:
    model = YOLO(str(ROOT / "configs/models/da-sfnet.yaml"))
    model.model.eval()
    with torch.no_grad():
        outputs = model.model(torch.zeros(1, 3, 640, 640))
    print(f"DA-SFNet constructed successfully; output type: {type(outputs).__name__}")


if __name__ == "__main__":
    main()

