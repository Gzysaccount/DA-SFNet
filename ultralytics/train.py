import warnings

warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('cfg/models/11/yolo11.yaml')
    model.train(data='datasets/FloorPlanCAD/data.yaml',
                cache=False,
                imgsz=640,
                epochs=200,
                batch=8,
                close_mosaic=10,
                device='0',
                optimizer='SGD',  # using SGD
                project='runs/train',
                name='exp',
                )
