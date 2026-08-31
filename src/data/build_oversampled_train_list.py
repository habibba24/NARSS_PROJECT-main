"""Build a train image-list .txt that oversamples rare-class images.

Ultralytics YOLO accepts a .txt file (one image path per line) as the
`train:` value in data.yaml, and duplicate lines are trained on as
duplicate samples - this is the standard way to rebalance a bbox-only
dataset (no segmentation masks here, so copy_paste augmentation isn't
usable). Band disease (class 0) and White Pox Disease (class 4) are
under-represented (201 and 84 of 2067 train images vs. 705-807 for the
common classes - see class instance counts), so their images are
repeated extra times in the list. Everything else appears once, same
as normal training.
"""
from pathlib import Path

DATASET_ROOT = Path(
    "C:/Users/Delta Tech/MAFFN_YOLOv5-based-Coral-reefs-health-detection-and-coral-reefs-dataset/"
    "MAFFN_YOLOv5-based-Coral-reefs-health-detection-and-coral-reefs-dataset-main/"
    "Dataset with annotation in YOLO format"
)
LABELS_DIR = DATASET_ROOT / "train" / "labels"
IMAGES_DIR = DATASET_ROOT / "train" / "images"
OUT_PATH = Path(__file__).resolve().parents[2] / "data" / "coral_health_train_oversampled.txt"

# extra copies added ON TOP OF the one guaranteed occurrence, keyed by class idx
EXTRA_COPIES = {0: 2, 4: 3}  # Band disease x3 total, White Pox Disease x4 total


def classes_in_label_file(path: Path) -> set[int]:
    classes = set()
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            classes.add(int(line.split()[0]))
    return classes


def main():
    if not LABELS_DIR.is_dir():
        print(
            "Disease detection is out of scope for this build (see README): the MAFFN\n"
            "YOLO disease dataset is not present on this machine.\n"
            f"  expected at: {DATASET_ROOT}\n"
            "Nothing to do - skipping oversampled-train-list generation."
        )
        return

    lines = []
    rare_hits = {c: 0 for c in EXTRA_COPIES}

    for label_path in sorted(LABELS_DIR.glob("*.txt")):
        image_path = IMAGES_DIR / f"{label_path.stem}.jpg"
        if not image_path.exists():
            matches = list(IMAGES_DIR.glob(f"{label_path.stem}.*"))
            if not matches:
                print(f"WARN: no image for {label_path.name}, skipping")
                continue
            image_path = matches[0]

        classes = classes_in_label_file(label_path)
        copies = 1
        for cls_idx, extra in EXTRA_COPIES.items():
            if cls_idx in classes:
                copies += extra
                rare_hits[cls_idx] += 1

        lines.extend([str(image_path)] * copies)

    OUT_PATH.write_text("\n".join(lines) + "\n")
    print(f"Wrote {len(lines)} lines ({len(lines) - sum(1 for _ in LABELS_DIR.glob('*.txt'))} extra) to {OUT_PATH}")
    for cls_idx, count in rare_hits.items():
        print(f"  class {cls_idx}: {count} source images x{1 + EXTRA_COPIES[cls_idx]} copies each")


if __name__ == "__main__":
    main()
