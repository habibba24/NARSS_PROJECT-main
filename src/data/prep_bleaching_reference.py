"""Build the reference data the bleaching module needs.

Two independent parts:

1. Per-genus "normal color" reference (runs now, no external data needed):
   crop each coral_soft bounding box, convert to CIE Lab (perceptually
   uniform, so "distance toward white" is meaningful), and record the
   mean/std Lab color per genus. bleaching_module.py compares a detected
   colony's pixels against its genus's reference to score paleness,
   instead of using one fixed "distance from white" threshold for every
   species (some genera are naturally pale).

2. Kaggle healthy/bleached calibration manifest (needs external data):
   scans data/external/kaggle_bleaching for class-labeled image folders
   and writes a manifest used later to tune the whiteness threshold and
   report precision/recall (see plan's Evaluation section). Prints setup
   instructions and exits cleanly if the folder is empty.
"""
import json
import os
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
CORAL_SOFT_YOLO = REPO_ROOT / "data" / "processed" / "coral_soft"
KAGGLE_DIR = REPO_ROOT / "data" / "external" / "kaggle_bleaching"
OUT_DIR = REPO_ROOT / "data" / "processed" / "bleaching_reference"


def build_genus_color_reference():
    data_yaml_names = eval(
        (CORAL_SOFT_YOLO / "data.yaml").read_text().splitlines()[-1].split("names: ", 1)[1]
    )
    class_names = data_yaml_names

    lab_samples: dict[str, list[np.ndarray]] = {name: [] for name in class_names}

    for split in ("train", "val"):
        img_dir = CORAL_SOFT_YOLO / "images" / split
        lbl_dir = CORAL_SOFT_YOLO / "labels" / split
        for label_path in lbl_dir.glob("*.txt"):
            image_path = img_dir / f"{label_path.stem}.jpg"
            if not image_path.exists():
                continue
            image_bgr = cv2.imread(str(image_path))
            if image_bgr is None:
                continue
            h, w = image_bgr.shape[:2]
            lab_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)

            for line in label_path.read_text().splitlines():
                if not line.strip():
                    continue
                parts = line.split()
                cls_idx = int(parts[0])
                cx, cy, bw, bh = (float(v) for v in parts[1:5])
                x0 = int(max(0, (cx - bw / 2) * w))
                x1 = int(min(w, (cx + bw / 2) * w))
                y0 = int(max(0, (cy - bh / 2) * h))
                y1 = int(min(h, (cy + bh / 2) * h))
                if x1 <= x0 or y1 <= y0:
                    continue
                crop = lab_image[y0:y1, x0:x1].reshape(-1, 3)
                # subsample so one huge box doesn't dominate the running stats
                if len(crop) > 2000:
                    idx = np.random.choice(len(crop), 2000, replace=False)
                    crop = crop[idx]
                lab_samples[class_names[cls_idx]].append(crop)

    reference = {}
    for name, samples in lab_samples.items():
        if not samples:
            continue
        stacked = np.concatenate(samples, axis=0).astype(np.float64)
        reference[name] = {
            "mean_lab": stacked.mean(axis=0).tolist(),
            "std_lab": stacked.std(axis=0).tolist(),
            "n_pixels": int(len(stacked)),
        }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "genus_colors.json"
    out_path.write_text(json.dumps(reference, indent=2))
    print(f"Wrote per-genus color reference for {len(reference)} classes to {out_path}")
    for name, stats in reference.items():
        print(f"  {name}: mean Lab={[round(v, 1) for v in stats['mean_lab']]} n={stats['n_pixels']}")


def build_kaggle_calibration_manifest():
    if not KAGGLE_DIR.exists() or not any(KAGGLE_DIR.iterdir()):
        print(
            f"\nNo data found in {KAGGLE_DIR}.\n"
            "To enable bleaching-threshold calibration/evaluation:\n"
            "  1. Download 'Healthy and Bleached Corals Image Classification' from Kaggle\n"
            "     (kaggle.com/datasets/vencerlanz09/healthy-and-bleached-corals-image-classification)\n"
            f"  2. Extract it into {KAGGLE_DIR}\n"
            "  3. Re-run this script.\n"
        )
        return

    rows = []
    # os.walk with followlinks=True (not Path.rglob, which doesn't recurse into
    # symlinked directories) since the dataset folders are symlinked in rather
    # than copied, to avoid duplicating ~900 images on an already-tight disk.
    for dirpath, _, filenames in os.walk(KAGGLE_DIR, followlinks=True):
        parent_name = Path(dirpath).name.lower()
        if "bleach" in parent_name:
            label = "bleached"
        elif "healthy" in parent_name:
            label = "healthy"
        else:
            continue
        for filename in filenames:
            if Path(filename).suffix.lower() in {".jpg", ".jpeg", ".png"}:
                rows.append((str(Path(dirpath) / filename), label))

    if not rows:
        print(f"Found files in {KAGGLE_DIR} but none matched expected 'healthy'/'bleached' folder naming.")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = OUT_DIR / "kaggle_calibration_manifest.csv"
    with open(manifest_path, "w") as f:
        f.write("image_path,label\n")
        for path, label in rows:
            f.write(f"{path},{label}\n")
    print(f"Wrote {len(rows)} calibration images to {manifest_path}")


if __name__ == "__main__":
    build_genus_color_reference()
    build_kaggle_calibration_manifest()
