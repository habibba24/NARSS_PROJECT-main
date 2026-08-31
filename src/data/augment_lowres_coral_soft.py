"""Augment colony_detector's training set with resolution-degraded copies.

Direct response to a diagnosed failure: the colony detector, trained only
on coral_soft's large (3000-4000px) aquarium photos, produces zero
detections on the same image once downscaled to ~300px (confirmed via a
controlled downscale test on a val image - 1 box at full res, 0 after
downscaling). It never saw what a genuinely low-detail source photo
looks like during training.

This generates a second version of every training image: downscaled to
a small size roughly matching the failure case, then upscaled back with
bilinear interpolation (baking in the blur/detail loss a real
low-resolution source photo has), added as an ADDITIONAL training
sample with the same YOLO labels - normalized box coordinates are
resolution-independent, so labels transfer unchanged. Validation images
are left untouched; they should still reflect normal/clean conditions
so accuracy numbers stay comparable to earlier runs.
"""
import random
from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
TRAIN_IMG_DIR = REPO_ROOT / "data" / "processed" / "coral_soft" / "images" / "train"
TRAIN_LBL_DIR = REPO_ROOT / "data" / "processed" / "coral_soft" / "labels" / "train"
MIN_DEGRADED_LONG_SIDE = 250
MAX_DEGRADED_LONG_SIDE = 450
SEED = 42


def main():
    random.seed(SEED)
    image_paths = sorted(TRAIN_IMG_DIR.glob("*.jpg"))
    # only degrade originals, not copies from a previous run of this script
    image_paths = [p for p in image_paths if not p.stem.endswith("_lowres")]
    print(f"Generating low-resolution-augmented copies for {len(image_paths)} training images...")

    written = 0
    for image_path in image_paths:
        label_path = TRAIN_LBL_DIR / f"{image_path.stem}.txt"
        if not label_path.exists():
            continue

        im = Image.open(image_path).convert("RGB")
        orig_size = im.size
        target_long_side = random.randint(MIN_DEGRADED_LONG_SIDE, MAX_DEGRADED_LONG_SIDE)
        scale = target_long_side / max(orig_size)
        small_size = (max(1, int(orig_size[0] * scale)), max(1, int(orig_size[1] * scale)))

        degraded = im.resize(small_size, Image.BILINEAR).resize(orig_size, Image.BILINEAR)

        out_stem = f"{image_path.stem}_lowres"
        degraded.save(TRAIN_IMG_DIR / f"{out_stem}.jpg", "JPEG", quality=85)
        (TRAIN_LBL_DIR / f"{out_stem}.txt").write_text(label_path.read_text())
        written += 1

    total = len(list(TRAIN_IMG_DIR.glob("*.jpg")))
    print(f"Wrote {written} low-resolution-augmented image+label pairs")
    print(f"Training set now has {total} images total (was {total - written})")


if __name__ == "__main__":
    main()
