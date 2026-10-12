# MAR20 YOLO Benchmark

A controlled comparison of five nano-scale YOLO detectors — **YOLOv5nu, YOLOv8n, YOLO11n, YOLO12n and YOLO26n** — for fine-grained aircraft-type detection on the [MAR20](https://gcheng-nwpu.github.io/) remote-sensing dataset (20 classes, A1–A20).

All models are trained with one identical configuration on one class-stratified split, evaluated once on a held-out test set, and then compared with statistical tests (bootstrap CIs, McNemar / Cochran's Q, RM-ANOVA / Friedman) and Grad-CAM.

## Repository layout

```
.
├── README.md
├── requirements.txt
├── code/
│   ├── mar20_baseline.py      # data preparation, training and test evaluation (all 5 models)
│   └── analysis/              # statistical and Grad-CAM analysis
│       ├── common.py          # GT/prediction loading, matching, mAP helpers
│       ├── val_all.py         # re-run test evaluation and save predictions (COCO JSON)
│       ├── check.py           # recompute mAP from saved predictions (sanity check)
│       ├── capture.py         # store per-image matched predictions for every model
│       ├── stats.py           # bootstrap CIs, McNemar, Cochran's Q, RM-ANOVA, Friedman
│       ├── gradcam.py         # Grad-CAM for Ultralytics YOLO detection heads
│       ├── cam_run.py         # quantitative Grad-CAM (in-box energy, pointing game)
│       ├── cam_fig.py, cam_fig2.py, qual_fig.py, samples_fig.py, figs.py   # figures
│       └── tables.py          # LaTeX tables
└── results/
    ├── mar20_test_metrics.json   # test metrics + per-class AP50 / AP50-95 for all models
    ├── analysis/                 # stats.json, cam_metrics.json, cam_stats.json
    └── runs_mar20/
        ├── <model>/              # training curves, results.csv, args.yaml, weights/best.pt
        └── test_<model>/         # test PR/F1 curves, confusion matrices, sample predictions
```

Not included (created at run time): `MAR20/` (raw dataset) and `mar20_yolo/` (converted YOLO dataset).

## Setup

```bash
pip install -r requirements.txt
```

Install a CUDA build of PyTorch for GPU training (see [pytorch.org](https://pytorch.org/get-started/locally/)). The pretrained `*.pt` weights are downloaded automatically by Ultralytics.

### Dataset

Download MAR20 and extract it to `MAR20/` in the repository root (next to `code/`). The script finds the folders by name, so the expected structure is:

```
MAR20/
├── Annotations/Horizontal Bounding Boxes/*.xml
├── ImageSets/Main/{train,test}.txt
└── JPEGImages/*.jpg
```

Only the horizontal bounding boxes (HBB) are used.

## Usage

Prepare the data, train all five models and evaluate them on the test split:

```bash
python code/mar20_baseline.py
```

The paths are relative to the repository root, so the script can run from any directory.

What the script does:

1. **Data preparation** – converts VOC XML to YOLO format, fixes `<size>` fields with 0×0 values from the image, and makes a class-stratified **7:2:1** split of all 3,842 images (seed 0). It also runs a near-duplicate (dHash) check between the splits and writes `mar20_yolo/data.yaml`.
2. **Training** – trains every model with the same settings (200 epochs, 640 px, batch 16, seed 0, deterministic, AMP) into `results/runs_mar20/<model>/`.
3. **Testing** – evaluates `best.pt` of every model once on the test split, writes plots to `results/runs_mar20/test_<model>/`, and writes all metrics plus per-class AP to `results/mar20_test_metrics.json`.

Useful options:

| Option | Description |
| --- | --- |
| `--models 8 11` | Train/test only the selected versions (`5 8 11 12 26`) |
| `--scale s` | Model scale: `n` (default), `s`, `m`, `l`, `x` |
| `--split official` | Keep the official MAR20 test list; validation is taken from the official train list (`--val-frac`) |
| `--split 8:1:1` | Custom train:val:test ratio (default `7:2:1`) |
| `--epochs`, `--batch`, `--imgsz`, `--seed`, `--workers` | Training hyper-parameters |
| `--only-prepare` | Only build `mar20_yolo/` |
| `--force-prepare` | Rebuild `mar20_yolo/` even if it exists |
| `--skip-train` | Only evaluate existing `best.pt` files (e.g. the released weights in `results/runs_mar20/`) |
| `--mar20-root PATH` | Location of the raw dataset (default `MAR20/`) |

## Results

Test split: 381 images, 2,241 objects, 20 classes.

| Model | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 | Train time (min) |
| --- | ---: | ---: | ---: | ---: | ---: |
| YOLOv5nu | 95.17 | 94.51 | 97.19 | 79.78 | 74.5 |
| YOLOv8n | 96.04 | 93.63 | 97.21 | 79.77 | 77.1 |
| YOLO11n | **96.24** | 94.36 | 97.06 | 80.04 | 79.7 |
| YOLO12n | 95.50 | **95.00** | **97.29** | **80.45** | 110.7 |
| YOLO26n | 95.42 | 93.60 | 96.96 | 80.28 | 99.0 |

Main findings:

- All models are within 0.33 (mAP@0.5) and 0.68 (mAP@0.5:0.95) percentage points of each other.
- The image-level bootstrap 95% CIs of all pairwise differences include zero.
- RM-ANOVA and Friedman tests find no model effect on per-class AP or per-image F1.
- McNemar tests (Holm-corrected) find one significant pair (YOLO12n vs. YOLO26n) at the F1-optimal thresholds, and none at a common threshold of 0.25.
- Grad-CAM puts 25.4–53.3% of its energy on the target aircraft, but this focus does not follow accuracy.
- A11 and A13 are the most difficult classes. Most errors are wrong type labels, not missed objects.

## Analysis scripts

The scripts in `code/analysis/` reproduce the statistical tests, Grad-CAM measurements, figures and tables. They need the trained weights in `results/runs_mar20/` and the converted dataset in `mar20_yolo/`. The precomputed outputs are in `results/analysis/`.

Typical order:

```bash
python val_all.py      # test predictions -> val_cpu/
python capture.py      # per-image matches -> capture.pkl
python stats.py        # -> stats.json
python cam_run.py      # -> cam_metrics.json
python tables.py       # -> LaTeX tables
```

> **Note:** Some scripts still have hard-coded absolute paths from the original environment (for example `/home/claude/work` and `/home/claude/paper/figs`). Change them to your local paths before you run the scripts.

## Citation

If you use this code, please cite:

> E. Dogruluk and I. Aruk, "Nano YOLO models for fine-grained aircraft recognition in remote sensing images: A statistical benchmark," *submitted to Pattern Analysis and Applications*.

## Acknowledgements

- MAR20 dataset: W. Yu, G. Cheng, M. Wang, Y. Yao, X. Xie, X. Yao and J. Han, "MAR20: A benchmark for military aircraft recognition in remote sensing images," *National Remote Sensing Bulletin*, 27(12):2688–2696, 2023. doi:[10.11834/jrs.20222139](https://doi.org/10.11834/jrs.20222139)
- [Ultralytics](https://github.com/ultralytics/ultralytics) for the YOLO implementations.
