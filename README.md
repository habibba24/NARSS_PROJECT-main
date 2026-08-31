# Coral Reef Early-Damage Detection

Detects two damage patterns on raw underwater coral photos — algae
overgrowth and bleaching — and outputs box/mask-level findings per
photo. See `plans` (or ask for the original plan file) for the full
design rationale; this README covers setup and how to run each stage.

A disease detector was originally planned as a third damage type, meant
to be fine-tuned on an external Roboflow dataset. That dataset turned
out to only have coral genus/species labels (Acropora, Porites,
Sarcophyton, ...), not disease labels, and no other labeled disease
source was available, so **disease detection was dropped from scope**.
`src/models/disease_detector.py` and `src/data/prep_disease_external.py`
no longer exist in this repo for that reason - if disease detection
becomes worth revisiting, it needs an actual disease-labeled dataset
sourced first.

## Why the pipeline is shaped the way it is

- **No damage labels exist in the source data.** `coral_soft` (bbox,
  healthy coral genus photos) and `archive-2` (sparse point labels,
  substrate/genus composition) don't contain a single bleaching or
  damage-region annotation. Each remaining damage type is bootstrapped
  differently:
  - **Algae** — grown automatically from `archive-2`'s algae/macroalgae/
    green_fleshy_algae points into weak masks (`grow_algae_masks.py`).
    CCA and turf are deliberately excluded (not damage indicators).
  - **Bleaching** — not a trained model at all. It's a deterministic
    color-space calculation: how far has a detected colony's color moved
    from its own genus's normal appearance toward white. Calibrated
    against a public Kaggle healthy/bleached set (923 images).
- **Preprocessing lives inside the exported model**, not a separate
  script, because the deployment target is a website/mobile app with no
  Python backend. Concretely: underwater color-cast correction
  (gray-world white balance) and normalization are real tensor ops
  embedded in every exported `.onnx` graph. Resizing an arbitrary photo
  to the network's fixed input size happens once, client-side, before
  the tensor is built — this is a real ONNX-tracing constraint (dynamic
  resize doesn't trace into a portable static graph), not a shortcut,
  and it's exactly how every deployed vision model handles arbitrary
  input resolutions.
- **Each stage exports as its own `.onnx` file**, not one monolithic
  graph for the whole pipeline. Running colony detection, then cropping
  to each detected box, then running algae/bleaching checks on that
  crop, is Python-level control flow that doesn't trace into a single
  static graph. A thin orchestration layer (the equivalent of
  `CoralDamagePipeline` in `src/pipeline/coral_damage_model.py`, ~50
  lines) needs to be reimplemented in JS/Swift/Kotlin at the app layer
  to sequence calls between the three `.onnx` files. Each individual
  file's internal preprocessing is still fully embedded — this only
  affects cross-stage orchestration, not the preprocessing requirement
  itself.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Apple Silicon (M1/M2/...) trains via PyTorch's MPS backend automatically
(`--device mps`); no CUDA GPU is required or used.

## Data layout

Raw datasets are **not** in the repo (`.gitignore`). Download each one and
place it at the exact path below — every script resolves paths relative to
the repo root, so no code edits are needed if the folders match.

| Dataset | Put it here | Expected contents | Used by |
|---|---|---|---|
| **coral_soft** — bbox JSON, healthy-coral genus photos | `coral_soft/` | `annotations/*.json` and `image/<Genus>/*.JPG` | `src/data/coral_soft_to_yolo.py`, `src/data/prep_bleaching_reference.py` |
| **archive-2** — CoralNet-style point labels (folder may be named `archive (1)`, `archive-2`, or `archive-2-subset`; first match wins) | `archive (1)/` | `combined_annotations_remapped.csv` and `images/images/*.jpg` | `src/data/grow_algae_masks.py`, `src/data/build_algae_tiles.py` |
| **Kaggle "Healthy and Bleached Corals Image Classification"** (923 imgs) | `data/external/kaggle_bleaching/` | `healthy_corals/*` and `bleached_corals/*` | `src/data/prep_bleaching_reference.py`, `src/eval/calibrate_bleaching_threshold.py` |
| **Coralscapes** (Cityscapes-style) — reef path model + eval masks | `data/external/coralscapes/` *or* set `CORALSCAPES_ROOT` env var (e.g. `D:/coralscapes/coralscapes`) | `leftImg8bit/{train,val}/…`, `gtFine/{train,val}/…`, `classes.json` | `src/data/coralscapes_to_yolo_seg.py`, `src/data/coralscapes_to_yolo_det.py`, `src/eval/evaluate_all.py` |
| **MAFFN YOLOv5 coral-health disease set** — *out of scope, optional* | any path; edit `DATASET_ROOT` at the top of `src/data/build_oversampled_train_list.py` | `train/{images,labels}/…` in YOLO format | `src/data/build_oversampled_train_list.py` only |

Everything under `data/processed/` is **generated** by the data-prep step
below (also git-ignored) — don't create it by hand:

| Generated folder | Produced by |
|---|---|
| `data/processed/coral_soft/` | `src/data/coral_soft_to_yolo.py` |
| `data/processed/algae_tiles/` | `src/data/build_algae_tiles.py` |
| `data/processed/algae_seg/` | `src/data/coralscapes_to_yolo_seg.py` |
| `data/processed/coral_reef_det/` | `src/data/coralscapes_to_yolo_det.py` |
| `data/processed/bleaching_reference/` | `src/data/prep_bleaching_reference.py` |

### Trained checkpoints

`models/checkpoints/` **is committed** (colony_detector, algae_classifier,
algae_segmenter — `weights/best.pt` + `weights/last.pt` + training curves,
~82 MB total). Clone the repo and you can run eval / export / inference
immediately, or resume a training run from `last.pt`, without retraining
from scratch. The `data/*.yaml` files `coral_health_augmented.yaml` and
`smoke_coral_health.yaml` still carry absolute paths and belong to the
out-of-scope disease work — ignore them unless you're reviving that.

## Two paths

The damage checks run in one of two modes depending on the photo:

- **reef** (`CoralDamagePipeline.run_reef`, the deployment target) — a single
  semantic-segmentation pass with the pretrained **Coralscapes DINOv3**
  model (`EPFL-ECEO/coralscapes-vit-b-dpt`) reads coral / bleached-coral /
  algae straight off the frame. No colony detector, no colour module.
  Needs `huggingface-cli login` once (the DINOv3 backbone is a gated Meta
  repo). On Coralscapes val: algae IoU 0.56 (R 0.86), bleached-coral IoU
  0.70 (R 0.92).
- **aquarium** (`CoralDamagePipeline.run`) — the original stack: a YOLO
  **colony detector** (trained on `coral_soft`, mAP@50 ~0.86) crops each
  colony, then a deterministic colour-space **bleaching** calc (per-genus
  reference, F1 ~0.54) and the DINOv3 model run per crop. For tank/close-up
  coral shots; it finds nothing on wide reef scenes.

Earlier attempts at the algae stage that did **not** work and are kept only
for reference: weak flood-fill masks from CoralNet points → YOLO-seg (mask
mAP50 ~0.01, `grow_algae_masks.py` + `algae_segmenter` history), real
Coralscapes polygons → YOLO-seg (~0.03), a 224px tile classifier
(`algae_classifier.py`, acc 0.82 — the lightweight non-ViT fallback), and a
reef "coral" detector (`coralscapes_to_yolo_det.py`, mAP50 ~0.34).

## Pipeline, in order

```bash
export PYTHONPATH=src
export HF_TOKEN=hf_...          # or: huggingface-cli login  (reef path only)

# 1. Data prep
python src/data/coral_soft_to_yolo.py            # aquarium colony detector + bleaching ref
python src/data/augment_lowres_coral_soft.py
python src/data/prep_bleaching_reference.py      # needs data/external/kaggle_bleaching/ ; skips cleanly if absent
python src/data/coralscapes_to_yolo_seg.py       # reef algae eval masks (needs Coralscapes)

# 2. Train the aquarium colony detector (reef path is pretrained, nothing to train)
python src/models/colony_detector.py --epochs 150 --device 0

# 3. Evaluate (reef segmenter + aquarium colony detector + aquarium bleaching)
python src/eval/evaluate_all.py

# 4. Export the aquarium stages to ONNX (the reef DINOv3 ViT is not exported yet)
python src/export/export_onnx.py

# 5. Demo end-to-end on a raw photo
python src/inference/run_demo.py path/to/reef_photo.jpg               # --domain reef (default)
python src/inference/run_demo.py path/to/tank_photo.jpg --domain aquarium
python src/inference/run_demo.py path/to/photo.jpg --mode onnx_proof --stage colony_detector
```

## Scope note

This build produces a working, runnable pipeline that trains and exports
correctly end-to-end — not fully-converged production-quality models.
Running the training commands above with realistic epoch counts
(potentially hours on this CPU/MPS-only machine) is a follow-up step,
not something already done here.

## Disk note

Only ~8-10GB was free on this machine when this was built. `data/processed/`
re-encodes `coral_soft` (~1.1GB) but symlinks `archive-2` and the Kaggle
bleaching set rather than copying them. Keep an eye on `df -h` before
increasing epoch counts or image sizes materially.
