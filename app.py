"""Gradio demo for DA-SFNet CAD drawing object detection."""

from functools import lru_cache
from pathlib import Path
import sys
import time

import gradio as gr
import numpy as np


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from ultralytics import YOLO  # noqa: E402


DEFAULT_WEIGHTS = ROOT / "weights" / "da-sfnet-best.pt"


@lru_cache(maxsize=2)
def load_model(weights: str) -> YOLO:
    path = Path(weights).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return YOLO(str(path))


def detect(image, confidence: float, iou: float, image_size: int, device: str, weights: str):
    if image is None:
        raise gr.Error("Please upload a CAD drawing image.")
    try:
        model = load_model(weights)
        started = time.perf_counter()
        result = model.predict(
            source=image,
            conf=float(confidence),
            iou=float(iou),
            imgsz=int(image_size),
            device=device.strip(),
            max_det=300,
            verbose=False,
        )[0]
        elapsed_ms = (time.perf_counter() - started) * 1000.0
    except Exception as exc:
        raise gr.Error(f"Inference failed: {exc}") from exc

    # Ultralytics returns a BGR visualization; Gradio expects RGB.
    annotated = result.plot()
    annotated = annotated[..., ::-1]
    rows = []
    if result.boxes is not None:
        xyxy = result.boxes.xyxy.detach().cpu().numpy()
        scores = result.boxes.conf.detach().cpu().numpy()
        classes = result.boxes.cls.detach().cpu().numpy().astype(int)
        for box, score, class_id in zip(xyxy, scores, classes):
            x1, y1, x2, y2 = box.tolist()
            rows.append(
                [
                    result.names[class_id],
                    class_id,
                    round(float(score), 4),
                    round(x1, 1),
                    round(y1, 1),
                    round(x2, 1),
                    round(y2, 1),
                ]
            )
    summary = f"**Detections:** {len(rows)} &nbsp; | &nbsp; **End-to-end time:** {elapsed_ms:.1f} ms"
    return np.ascontiguousarray(annotated), rows, summary


def build_demo() -> gr.Blocks:
    examples = [str(path) for path in sorted((ROOT / "data/FloorPlanCAD/images/test").glob("*"))[:6]]
    with gr.Blocks(title="DA-SFNet CAD Drawing Detector", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# DA-SFNet: CAD Drawing Object Detection\n"
            "Upload a CAD floor-plan image to test the released DA-SFNet checkpoint. "
            "The detector contains the **DA-FFT** and **MST-LADA** modules described in the paper."
        )
        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(type="numpy", label="Input CAD drawing")
                weights = gr.Textbox(value=str(DEFAULT_WEIGHTS), label="Checkpoint")
                with gr.Row():
                    confidence = gr.Slider(0.01, 0.95, value=0.25, step=0.01, label="Confidence")
                    iou = gr.Slider(0.10, 0.95, value=0.70, step=0.05, label="NMS IoU")
                with gr.Row():
                    image_size = gr.Dropdown([320, 480, 640, 800, 960], value=640, label="Image size")
                    device = gr.Textbox(value="", label="Device", placeholder="0, cpu, or leave blank")
                run_button = gr.Button("Run detection", variant="primary")
            with gr.Column(scale=1):
                output_image = gr.Image(type="numpy", label="Detection result")
                summary = gr.Markdown()
        detections = gr.Dataframe(
            headers=["class", "class_id", "confidence", "x1", "y1", "x2", "y2"],
            datatype=["str", "number", "number", "number", "number", "number", "number"],
            interactive=False,
            label="Detections",
        )
        if examples:
            gr.Examples(examples=examples, inputs=input_image, label="FloorPlanCAD test examples")
        run_button.click(
            detect,
            inputs=[input_image, confidence, iou, image_size, device, weights],
            outputs=[output_image, detections, summary],
        )
    return demo


if __name__ == "__main__":
    build_demo().launch(server_name="127.0.0.1", server_port=7860, inbrowser=True)

