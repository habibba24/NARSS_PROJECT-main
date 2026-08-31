"""One summary table for the three trained/deterministic components' accuracy
(colony detector, algae segmenter, bleaching module) - see the plan's
Evaluation & accuracy metrics section for why each needs its own metric
rather than a single "accuracy %". Each block is independent and skips
cleanly (printing why) if its checkpoint/data isn't available yet, so
this can be run at any point during the build instead of only at the end.
"""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
results_table = []  # (component, metric, value, n)


def eval_colony_detector():
    from models import colony_detector

    try:
        model = colony_detector.load_best()
    except FileNotFoundError as e:
        print(f"[colony_detector] skipped: {e}")
        return
    metrics = model.val(data=str(colony_detector.DATA_YAML), verbose=False)
    results_table.append(("colony_detector", "mAP50", metrics.box.map50, metrics.box.nc))
    results_table.append(("colony_detector", "mAP50-95", metrics.box.map, metrics.box.nc))
    results_table.append(("colony_detector", "recall(mean)", metrics.box.mr, metrics.box.nc))


def eval_algae_segmenter():
    import numpy as np
    from PIL import Image

    from models import algae_segmenter

    val_img = REPO_ROOT / "data" / "processed" / "algae_seg" / "images" / "val"
    val_msk = REPO_ROOT / "data" / "processed" / "algae_seg" / "masks" / "val"
    if not val_img.is_dir() or not any(val_img.iterdir()):
        print("[algae_segmenter] skipped: no data/processed/algae_seg/val "
              "(run src/data/coralscapes_to_yolo_seg.py)")
        return
    try:
        model = algae_segmenter.load_model()
    except Exception as e:  # gated repo / no hf login / offline
        print(f"[algae_segmenter] skipped: {type(e).__name__}: {str(e)[:120]}")
        return

    # pixel IoU / precision / recall for the 'algae covered substrate' class,
    # against the Coralscapes val masks (converted to binary by the prep script).
    inter = union = tp = fp = fn = 0
    n = 0
    for ip in sorted(val_img.glob("*.jpg")):
        mp = val_msk / f"{ip.stem}.png"
        if not mp.exists():
            continue
        gt = np.array(Image.open(mp)) > 127
        pred = algae_segmenter.segment_crop(model, np.array(Image.open(ip).convert("RGB")))["mask"]
        inter += int(np.logical_and(pred, gt).sum())
        union += int(np.logical_or(pred, gt).sum())
        tp += int(np.logical_and(pred, gt).sum())
        fp += int(np.logical_and(pred, ~gt).sum())
        fn += int(np.logical_and(~pred, gt).sum())
        n += 1

    iou = inter / union if union else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    results_table.append(("algae_segmenter", "pixel_IoU", iou, n))
    results_table.append(("algae_segmenter", "pixel_precision", precision, n))
    results_table.append(("algae_segmenter", "pixel_recall", recall, n))
    results_table.append(("algae_segmenter", "pixel_F1", f1, n))


def eval_bleaching_module():
    import csv

    import torch
    from PIL import Image

    from models import colony_detector
    from models.bleaching_module import BleachingModule
    from pipeline.coral_damage_model import CoralDamagePipeline

    manifest_path = REPO_ROOT / "data" / "processed" / "bleaching_reference" / "kaggle_calibration_manifest.csv"
    if not manifest_path.exists():
        print("[bleaching_module] skipped: no calibration manifest (Kaggle set not prepared)")
        return
    try:
        pipeline = CoralDamagePipeline()
    except FileNotFoundError as e:
        print(f"[bleaching_module] skipped: {e}")
        return

    tp = fp = tn = fn = 0
    with open(manifest_path) as f:
        for row in csv.DictReader(f):
            findings = pipeline.run(row["image_path"])
            predicted_bleached = any(f_.damage_type == "bleaching" for f_ in findings)
            actual_bleached = row["label"] == "bleached"
            if predicted_bleached and actual_bleached:
                tp += 1
            elif predicted_bleached and not actual_bleached:
                fp += 1
            elif not predicted_bleached and actual_bleached:
                fn += 1
            else:
                tn += 1

    n = tp + fp + tn + fn
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    results_table.append(("bleaching_module", "precision", precision, n))
    results_table.append(("bleaching_module", "recall", recall, n))
    results_table.append(("bleaching_module", "F1", f1, n))


def main():
    eval_colony_detector()
    eval_algae_segmenter()
    eval_bleaching_module()

    print("\n=== Summary ===")
    print(f"{'component':20s} {'metric':40s} {'value':>8s} {'N':>6s}")
    for component, metric, value, n in results_table:
        print(f"{component:20s} {metric:40s} {value:8.3f} {n:6d}")

    if not results_table:
        print("Nothing to evaluate yet - train at least one component first.")


if __name__ == "__main__":
    main()
