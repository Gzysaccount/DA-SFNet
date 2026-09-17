# DA-SFNet

Research-code package for **DA-SFNet: A Direction-Aware Spatial-Frequency Collaborative Enhancement Network for Architectural Object Detection in CAD Drawings**, a YOLO11-based detector for dense CAD floor-plan drawings. The model combines:

- **DA-FFT**: direction-adaptive local frequency decomposition and bounded residual injection for suppressing text/line noise while retaining weak structural targets.
- **MST-LADA**: multi-scale structure-tensor local directional attention for strengthening horizontal, vertical, and turning structures.

This repository contains the modified Ultralytics source, model definitions, the released checkpoint, reproducible train/validation/inference scripts, a Gradio test UI, the FloorPlanCAD data used in the experiments, and the retained dual-module experiment records.

## Repository layout

```text
DA-SFNet/
├── app.py                       # Gradio visualization UI
├── configs/models/da-sfnet.yaml
├── data/FloorPlanCAD/           # train/val/test images and YOLO labels
├── experiments/                 # retained training and held-out-test artifacts
├── scripts/                     # train, val, predict, export, checks
├── ultralytics/                 # vendored framework with DA-FFT/MST-LADA registration
├── weights/
│   ├── da-sfnet-best.pt         # released final checkpoint
│   └── yolo11n.pt               # initialization checkpoint
└── pyproject.toml
```

The custom modules are implemented in:

- `ultralytics/nn/modules/da_fft.py`
- `ultralytics/nn/modules/mst_lada.py`

They are registered in `ultralytics/nn/modules/__init__.py`, `ultralytics/nn/modules/block.py`, and `ultralytics/nn/tasks.py` so the supplied YAML and checkpoint can be loaded directly.

## Installation

Python 3.10 or 3.11 is recommended. Create an isolated environment and install PyTorch for your CUDA version first (or use the CPU wheel), then install this repository:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate

# Install a matching PyTorch build: https://pytorch.org/get-started/locally/
pip install -e .
pip install -r requirements-app.txt
```

For the large files, clone through Git LFS:

```bash
git lfs install
git clone <your-repository-url>
cd DA-SFNet
git lfs pull
```

To publish this prepared local repository to a new GitHub repository:

```bash
git add .
git commit -m "Initial DA-SFNet release"
git branch -M main
git remote add origin https://github.com/<your-account>/DA-SFNet.git
git push -u origin main
```

Replace the placeholder repository URL in `CITATION.cff` after creating the GitHub repository. Because the complete dataset is included, check both its redistribution terms and your Git LFS quota before the first push.

## Quick start

Validate the dataset package without importing PyTorch:

```bash
python scripts/check_dataset.py
```

Build the network and run a dummy forward pass:

```bash
python scripts/smoke_test.py
```

Run inference:

```bash
python scripts/predict.py --source data/FloorPlanCAD/images/test
```

Launch the visual test UI:

```bash
python app.py
```

Open `http://127.0.0.1:7860` if the browser does not open automatically.

## Training

The default command reproduces the released run settings recorded by Ultralytics: 200 epochs, 640-pixel input, batch size 8, SGD, initial learning rate 0.01, momentum 0.937, three warm-up epochs, seed 1, deterministic mode, and mosaic closure during the last 30 epochs.

```bash
python scripts/train.py --device 0
```

To change hardware or batch size:

```bash
python scripts/train.py --device 0 --batch 4 --workers 4
```

Outputs are written to `runs/train/`. The original machine-generated arguments and learning curves are retained under `experiments/train/`.

To reproduce the four-cell ablation matrix:

```bash
python scripts/train_ablation.py --device 0
```

## Evaluation

```bash
python scripts/val.py --split test --device 0
```

Held-out FloorPlanCAD test results reported in the paper:

| Configuration | MST-LADA | DA-FFT | mAP50 (%) | Peak F1 |
|---|:---:|:---:|---:|---:|
| Baseline |  |  | 54.9 | 0.52 |
| MST-LADA | ✓ |  | 56.5 | 0.54 |
| DA-FFT |  | ✓ | 56.9 | 0.53 |
| DA-SFNet | ✓ | ✓ | **61.9** | 0.53 |

For the final checkpoint, the held-out test metrics were precision 49.96%, recall 64.29%, mAP50 61.88%, and mAP50-95 48.83%.

## Export

```bash
python scripts/export.py --format onnx --device cpu
```

Other Ultralytics export targets can be selected through `--format`.

## Reproducibility notes

- Dataset configuration: `data/FloorPlanCAD/data.yaml`
- Final architecture: `configs/models/da-sfnet.yaml`
- Exact recorded run arguments: `experiments/train/da_sfnet_seed1_m30/args.yaml`
- Epoch-wise metrics: `experiments/train/*/results.csv`
- Held-out test plots: `experiments/test/`
- Consolidated paper metrics: `experiments/ablation_summary.csv`

The packaged `args.yaml` files are the authoritative records of the executed runs. Paths in those historical files reflect the original training machine; the scripts in this repository use portable relative paths.

## Data and licensing

The codebase is derived from Ultralytics and is distributed under the **GNU Affero General Public License v3.0**; see `LICENSE` and `NOTICE`.

FloorPlanCAD is third-party research data. Before redistributing or using it commercially, verify and comply with the dataset's original license and citation requirements. See `data/FloorPlanCAD/README.md`.

Large checkpoints and dataset images are configured for Git LFS. Confirm your GitHub LFS storage/bandwidth quota before pushing the complete data package.

## Citation

If this repository supports your work, cite the accompanying paper. The machine-readable author metadata is provided in `CITATION.cff`; update its repository URL and publication fields after the paper and GitHub repository are public.
