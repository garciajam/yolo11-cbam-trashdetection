# YOLO11 with CBAM

A modified build of [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics) that adds the **Convolutional Block Attention Module (CBAM)** to the network, developed for a thesis comparing a CBAM-augmented YOLO11m against the stock YOLO11m baseline on a custom object detection dataset.

## What this repo adds on top of stock YOLO11

- **`ultralytics/nn/modules/conv.py`** — `CBAM` module (channel attention + spatial attention, applied in sequence), wired into the model parser in `ultralytics/nn/tasks.py` so it can be dropped into any model YAML like a standard layer.
- **`yolo11m-CBAM.yaml`** — a YOLO11m variant with `CBAM` inserted after the backbone's `SPPF` output and after each of the three detection-head feature maps (P3/P4/P5) in the neck. The neck's downsampling convs are also swapped for `GhostConv` to help offset the extra attention parameters.
- **`yolo11m.yaml`** — the unmodified YOLO11m config, kept alongside the CBAM variant as the baseline for comparison.
- **`YOLO11OptimizedTraining.ipynb`** — training notebook that trains both variants (baseline and CBAM) with the same hyperparameters and automatically resumes from the last checkpoint if one exists.
- **`yolo11m.pt`** — pretrained YOLO11m weights used to seed baseline training/comparison.

Everything else under `ultralytics/` (engine, models, data pipeline, trainers, CLI, trackers, etc.) is the standard Ultralytics framework that CBAM is built on.

## Repository layout

```
ultralytics/            Vendored Ultralytics package (includes the CBAM module + parser changes)
docs/                    Ultralytics documentation source
examples/                Ultralytics usage examples
tests/                   Ultralytics test suite
docker/                  Ultralytics Dockerfiles
yolo11m.yaml             Baseline YOLO11m model config
yolo11m-CBAM.yaml        YOLO11m + CBAM model config
yolo11m.pt               Pretrained baseline weights
data.yaml                Sample dataset config (edit for your own dataset)
YOLO11OptimizedTraining.ipynb   Training notebook for both variants
```

## Setup

```bash
pip install ultralytics
```

Run scripts from the repository root. Python resolves the local `ultralytics/` package before the installed one, so the CBAM-enabled code in this repo is what actually gets used.

## Usage

### Train

```python
from ultralytics import YOLO

# CBAM-augmented model
model = YOLO("yolo11m-CBAM.yaml")
model.train(data="data.yaml", imgsz=960, epochs=300, batch=16)

# Baseline for comparison
model = YOLO("yolo11m.yaml")
model.train(data="data.yaml", imgsz=960, epochs=300, batch=16)
```

`YOLO11OptimizedTraining.ipynb` runs both configurations with matched hyperparameters and resumes automatically from `weights/last.pt` if training was interrupted.

### Validate / predict

```python
model = YOLO("path/to/best.pt")
model.val(data="data.yaml")
model.predict("path/to/image_or_video")
```

### Dataset format

`data.yaml` follows the standard Ultralytics dataset spec — `train`/`val`/`test` image directories plus `nc` and `names`. Point it at your own dataset before training; the file in this repo is a template and should be edited to match your data's paths and classes.

## Model configs

| Config | Backbone | Attention | Neck downsampling |
|---|---|---|---|
| `yolo11m.yaml` | YOLO11m | — | `Conv` |
| `yolo11m-CBAM.yaml` | YOLO11m | `CBAM` after SPPF and after each detection scale (P3/P4/P5) | `GhostConv` |

## Acknowledgements

- [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics) — base detection framework.
- Woo et al., ["CBAM: Convolutional Block Attention Module"](https://arxiv.org/abs/1807.06521) (ECCV 2018) — attention mechanism used in this modification.
