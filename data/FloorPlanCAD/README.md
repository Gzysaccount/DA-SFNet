# FloorPlanCAD data included for reproducibility

This directory contains the train, validation, and held-out test images and YOLO-format labels used by the DA-SFNet experiments. The detector uses 28 object classes listed in `data.yaml`.

FloorPlanCAD is third-party research data. Its inclusion here does not transfer ownership or replace the original dataset terms. Before publishing this repository, verify that redistribution is permitted and add the dataset's official homepage, license, and citation below. If redistribution is not permitted, remove `images/` and `labels/`, retain `data.yaml`, and provide an authorized download/preparation script instead.

Expected layout:

```text
FloorPlanCAD/
├── data.yaml
├── images/{train,val,test}/
└── labels/{train,val,test}/
```

Run `python scripts/check_dataset.py` from the repository root to verify image/label integrity.

