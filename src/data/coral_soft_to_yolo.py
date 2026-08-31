"""Convert coral_soft's bbox JSON annotations into a YOLO-format dataset.

coral_soft/annotations/*.json each contain: [{"image": "<name>.JPG",
"annotations": [{"label": str, "coordinates": {x, y, width, height}}]}]
where x,y are the box CENTER in pixels (verified against source images,
not top-left) and width/height are pixel box size. An image can carry
boxes for multiple coral classes.

Image files live under coral_soft/image/<Class>/<name>.JPG but are
inconsistently encoded (JPEG/PNG/MPO all under a .JPG extension), so
this script re-encodes every referenced image to real JPEG in the
output directory rather than copying bytes as-is.

Two corrections applied before saving, both confirmed necessary by
direct inspection (see the EXIF one - a Favosites annotation box
landed on an unrecognizable patch of encrusting growth without it, and
on a textbook, unmistakable Favosites colony with it):

1. EXIF orientation: 40 of 664 source images carry an EXIF rotation
   tag (their sensor-native pixel grid needs a 90 degree turn to match
   what any viewer or annotation tool actually displayed). The
   annotation coordinates were measured against that displayed,
   correctly-oriented image - loading raw pixels with plain
   Image.open() (as this script previously did) ignores that tag, so
   for those 40 images the stored width/height get used un-swapped and
   the box lands nowhere near the labeled coral. ImageOps.exif_transpose
   applies the tag's rotation before anything else touches the image.
2. Underwater color-cast correction (gray-world white balance): every
   real inference path (CoralDamagePipeline, the exported .onnx graphs)
   color-corrects its input before running the detector, but training
   images were previously left raw - a real train/inference distribution
   mismatch, not just a cosmetic difference. Applying the same
   correction here means the detector is finally trained on the same
   kind of input it's evaluated and deployed on.
"""
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

from pipeline.preprocessing import ColorCastCorrection

REPO_ROOT = Path(__file__).resolve().parents[2]
ANNOTATIONS_DIR = REPO_ROOT / "coral_soft" / "annotations"
IMAGE_DIR = REPO_ROOT / "coral_soft" / "image"
OUT_DIR = REPO_ROOT / "data" / "processed" / "coral_soft"
VAL_FRACTION = 0.2
SEED = 42


def find_image_file(image_name: str) -> Path | None:
    matches = list(IMAGE_DIR.glob(f"*/{image_name}"))
    if not matches:
        # extension case mismatch (e.g. .jpg vs .JPG) - fall back to stem search
        stem = Path(image_name).stem
        matches = [p for p in IMAGE_DIR.glob(f"*/{stem}.*")]
    return matches[0] if matches else None


def load_annotations():
    records = []
    labels = set()
    for json_path in sorted(ANNOTATIONS_DIR.glob("*.json")):
        with open(json_path) as f:
            entries = json.load(f)
        for entry in entries:
            image_name = entry["image"]
            boxes = entry["annotations"]
            image_path = find_image_file(image_name)
            if image_path is None:
                print(f"WARN: no image file found for {image_name} (from {json_path.name}), skipping")
                continue
            records.append({"image_path": image_path, "boxes": boxes})
            for box in boxes:
                labels.add(box["label"])
    return records, sorted(labels)


def convert_box_to_yolo(box, img_w, img_h):
    c = box["coordinates"]
    cx, cy, w, h = c["x"], c["y"], c["width"], c["height"]
    # clip to image bounds defensively
    cx = min(max(cx, 0), img_w)
    cy = min(max(cy, 0), img_h)
    w = min(w, img_w)
    h = min(h, img_h)
    return cx / img_w, cy / img_h, w / img_w, h / img_h


def load_corrected_rgb(image_path: Path, color_correction: ColorCastCorrection) -> Image.Image:
    """Open, apply EXIF rotation, then apply the same gray-world color
    correction every inference path uses - see module docstring for why
    both matter."""
    im = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
    tensor = torch.from_numpy(np.array(im)).permute(2, 0, 1).unsqueeze(0).float()
    corrected = color_correction(tensor)
    corrected = corrected.squeeze(0).permute(1, 2, 0).clamp(0, 255).byte().numpy()
    return Image.fromarray(corrected)


def main():
    random.seed(SEED)
    color_correction = ColorCastCorrection()
    records, class_names = load_annotations()
    class_index = {name: i for i, name in enumerate(class_names)}
    print(f"Found {len(records)} images, {len(class_names)} classes: {class_names}")

    random.shuffle(records)
    n_val = max(1, int(len(records) * VAL_FRACTION))
    splits = {"val": records[:n_val], "train": records[n_val:]}

    for split, split_records in splits.items():
        img_out = OUT_DIR / "images" / split
        lbl_out = OUT_DIR / "labels" / split
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for record in split_records:
            image_path = record["image_path"]
            try:
                im = load_corrected_rgb(image_path, color_correction)
            except Exception as e:
                print(f"WARN: failed to open {image_path}: {e}, skipping")
                continue
            img_w, img_h = im.size

            stem = image_path.stem
            out_image_path = img_out / f"{stem}.jpg"
            im.save(out_image_path, "JPEG", quality=90)

            lines = []
            for box in record["boxes"]:
                cx, cy, w, h = convert_box_to_yolo(box, img_w, img_h)
                cls_idx = class_index[box["label"]]
                lines.append(f"{cls_idx} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            (lbl_out / f"{stem}.txt").write_text("\n".join(lines) + "\n")

        print(f"{split}: wrote {len(split_records)} images")

    data_yaml = OUT_DIR / "data.yaml"
    yaml_lines = [
        f"path: {OUT_DIR.as_posix()}",
        "train: images/train",
        "val: images/val",
        f"nc: {len(class_names)}",
        f"names: {class_names}",
    ]
    data_yaml.write_text("\n".join(yaml_lines) + "\n")
    print(f"Wrote {data_yaml}")


if __name__ == "__main__":
    main()
