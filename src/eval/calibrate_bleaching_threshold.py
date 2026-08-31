"""Tune the bleaching module's two thresholds against the Kaggle
healthy/bleached set, instead of the untested guesses it shipped with
(per-pixel paling score > 0.5, colony flagged if > 5% of its pixels
cross that).

Colony detection is the expensive part - one model forward pass per
image - so it only runs once per image, for a handful of candidate
per-pixel thresholds at once. The per-colony fraction cutoff is then
swept cheaply afterward over the cached fractions (no re-running the
model), so trying many cutoff values is nearly free.

Only meaningful once the colony detector has had a real training run -
if it's still the 3-epoch mechanics-check checkpoint, most Kaggle photos
won't have a detected colony at all and every threshold will look
equally bad. See coral_damage_model.py's CoralDamagePipeline for what
these two numbers control.

Separate finding from a first run of this script: the colony detector's
confidence scores run low overall (top box on a training-distribution
validation image scored only ~0.20), so the 0.25 confidence cutoff
originally used here and in CoralDamagePipeline.run() was silently
producing zero detections almost everywhere. Both now default to 0.1,
picked by inspecting the actual confidence distribution, not tuned
rigorously - revisit if the detector gets more/better training data.
"""
import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "data" / "processed" / "bleaching_reference" / "kaggle_calibration_manifest.csv"
OUT_PATH = REPO_ROOT / "data" / "processed" / "bleaching_reference" / "threshold_calibration.csv"

PIXEL_THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7]
FRACTION_CUTOFFS = np.arange(0.0, 1.01, 0.02)
MIN_ACCEPTABLE_PRECISION = 0.5  # recall-over-precision per project decision, but not unusably low precision


def collect_scores():
    """One pass over every calibration image: run colony detection once,
    then for each detected colony compute (per pixel-threshold candidate)
    what fraction of its pixels are flagged pale. Returns a list of
    (actual_bleached: bool, max_fraction_per_pixel_threshold: list[float])
    - taking the max across colonies in an image, since one bleached
    colony is enough to call the photo bleached.
    """
    from pipeline.coral_damage_model import CoralDamagePipeline

    pipeline = CoralDamagePipeline()
    with open(MANIFEST_PATH) as f:
        manifest_rows = list(csv.DictReader(f))

    results = []
    for i, row in enumerate(manifest_rows):
        raw = np.array(Image.open(row["image_path"]).convert("RGB"))
        corrected = pipeline._correct_image(raw)
        colony_results = pipeline.colony_model.predict(Image.fromarray(corrected), conf=0.1, verbose=False)[0]
        genus_names = colony_results.names

        max_fraction = [0.0] * len(PIXEL_THRESHOLDS)
        for box in colony_results.boxes:
            x0, y0, x1, y1 = (float(v) for v in box.xyxy[0])
            genus = genus_names[int(box.cls[0])]
            if genus not in pipeline.bleaching_module.genus_names:
                continue
            crop = corrected[int(y0) : int(y1), int(x0) : int(x1)]
            if crop.size == 0:
                continue
            tensor = torch.from_numpy(crop).permute(2, 0, 1).unsqueeze(0).float()
            genus_idx = torch.tensor([pipeline.bleaching_module.genus_names.index(genus)])
            score = pipeline.bleaching_module(tensor, genus_idx)["paling_score"][0]
            for j, pt in enumerate(PIXEL_THRESHOLDS):
                frac = (score > pt).float().mean().item()
                max_fraction[j] = max(max_fraction[j], frac)

        results.append((row["label"] == "bleached", max_fraction))
        if (i + 1) % 100 == 0:
            print(f"  processed {i + 1}/{len(manifest_rows)}")

    return results


def sweep(rows):
    configs = []
    for j, pt in enumerate(PIXEL_THRESHOLDS):
        for fc in FRACTION_CUTOFFS:
            tp = fp = tn = fn = 0
            for actual_bleached, max_fraction in rows:
                predicted = max_fraction[j] > fc
                if predicted and actual_bleached:
                    tp += 1
                elif predicted and not actual_bleached:
                    fp += 1
                elif not predicted and actual_bleached:
                    fn += 1
                else:
                    tn += 1
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
            configs.append((pt, round(float(fc), 2), precision, recall, f1, tp, fp, tn, fn))
    return configs


def main():
    if not MANIFEST_PATH.exists():
        print(f"No calibration manifest at {MANIFEST_PATH} - run prep_bleaching_reference.py first")
        return

    print("Running colony detection + bleaching scoring once per calibration image...")
    rows = collect_scores()

    print("Sweeping fraction cutoffs (cheap - no model inference)...")
    configs = sweep(rows)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["pixel_threshold", "fraction_cutoff", "precision", "recall", "f1", "tp", "fp", "tn", "fn"])
        writer.writerows(configs)
    print(f"Wrote {len(configs)} configs to {OUT_PATH}")

    viable = [c for c in configs if c[2] >= MIN_ACCEPTABLE_PRECISION]
    pool = viable if viable else configs
    if not viable:
        print(f"WARNING: no config reaches precision >= {MIN_ACCEPTABLE_PRECISION}; picking best recall overall instead.")
    best = max(pool, key=lambda c: c[3])

    pt, fc, precision, recall, f1, *_ = best
    print(f"\nRecommended: pixel_threshold={pt}, fraction_cutoff={fc}")
    print(f"  -> precision={precision:.3f} recall={recall:.3f} f1={f1:.3f}")
    print(
        "\nTo apply: BleachingModule(threshold=<pixel_threshold>) and "
        "CoralDamagePipeline(bleaching_fraction_cutoff=<fraction_cutoff>)"
    )


if __name__ == "__main__":
    main()
