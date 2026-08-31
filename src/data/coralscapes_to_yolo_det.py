"""Build a reef-domain "coral region" detection dataset from Coralscapes.

The colony detector was trained on coral_soft (aquarium close-ups of 6
coral genera) and finds nothing on wide reef photos - which is the actual
deployment input and the domain the algae segmenter needs. This converts
Coralscapes' semantic masks into YOLO detection boxes for a single class,
"coral": connected components of the union of every coral class (alive,
bleached, and dead - a colony is worth localising regardless of state, so
the downstream algae/bleaching checks have a region to look inside).

Reads the Zenodo / Cityscapes-style layout (same as coralscapes_to_yolo_seg.py):
    <root>/leftImg8bit/<split>/<site>/<id>_leftImg8bit.png
    <root>/gtFine/<split>/<site>/<id>_gtFine.png     L mask, pixel = class id

Output:
    data/processed/coral_reef_det/images/<split>/<id>.jpg   raw (matches the reef imagery)
    data/processed/coral_reef_det/labels/<split>/<id>.txt   YOLO box: "0 cx cy w h"
    data/processed/coral_reef_det/data.yaml
"""
import json
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "data" / "processed" / "coral_reef_det"

_ROOT_CANDIDATES = [
    os.environ.get("CORALSCAPES_ROOT", ""),
    r"D:/coralscapes/coralscapes",
    r"D:/coralscapes",
    str(REPO_ROOT / "data" / "external" / "coralscapes"),
]
CORALSCAPES_ROOT = next(
    (Path(p) for p in _ROOT_CANDIDATES if p and (Path(p) / "leftImg8bit").is_dir()), None
)

# every coral class in Coralscapes (see classes.json): alive + bleached + dead
CORAL_CLASS_IDS = {3, 4, 6, 16, 17, 19, 20, 21, 22, 23, 25, 27, 28, 31, 32, 33, 34, 36, 37}
MIN_BOX_AREA_FRAC = 0.0008   # drop boxes smaller than this fraction of the image
MORPH_KERNEL = 7             # close gaps within a colony before component labelling
JPEG_QUALITY = 90
SPLITS = ("train", "val")


def mask_to_boxes(binary: np.ndarray) -> list[tuple[float, float, float, float]]:
    h, w = binary.shape
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_KERNEL, MORPH_KERNEL))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k)
    n, _, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    min_area = MIN_BOX_AREA_FRAC * h * w
    boxes = []
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area < min_area:
            continue
        cx, cy = (x + bw / 2) / w, (y + bh / 2) / h
        boxes.append((cx, cy, bw / w, bh / h))
    return boxes


def main():
    if CORALSCAPES_ROOT is None:
        raise SystemExit("Coralscapes not found - set CORALSCAPES_ROOT to the folder with leftImg8bit/ + gtFine/")
    classes = json.loads((CORALSCAPES_ROOT / "classes.json").read_text())
    coral_names = [n for n, i in classes.items() if i in CORAL_CLASS_IDS]
    print(f"Coralscapes root: {CORALSCAPES_ROOT}")
    print(f"coral = {len(CORAL_CLASS_IDS)} classes: {sorted(coral_names)}")

    lut = np.zeros(256, np.uint8)
    for cid in CORAL_CLASS_IDS:
        lut[cid] = 1

    stats = {}
    for split in SPLITS:
        images = sorted((CORALSCAPES_ROOT / "leftImg8bit" / split).rglob("*_leftImg8bit.png"))
        out_img = OUT_DIR / "images" / split
        out_lbl = OUT_DIR / "labels" / split
        out_img.mkdir(parents=True, exist_ok=True)
        out_lbl.mkdir(parents=True, exist_ok=True)

        n = with_box = total_boxes = 0
        for i, ip in enumerate(images):
            mp = ip.parents[3] / "gtFine" / split / ip.parent.name / ip.name.replace("_leftImg8bit.png", "_gtFine.png")
            if not mp.exists():
                print(f"WARN: no mask for {ip.name}, skipping")
                continue
            stem = ip.name.replace("_leftImg8bit.png", "")

            rgb = np.array(Image.open(ip).convert("RGB"))
            lab = np.array(Image.open(mp))
            if lab.shape != rgb.shape[:2]:
                lab = np.array(Image.fromarray(lab).resize((rgb.shape[1], rgb.shape[0]), Image.NEAREST))
            binary = lut[lab]

            boxes = mask_to_boxes(binary) if binary.any() else []
            Image.fromarray(rgb).save(out_img / f"{stem}.jpg", "JPEG", quality=JPEG_QUALITY)
            (out_lbl / f"{stem}.txt").write_text(
                "".join(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n" for cx, cy, bw, bh in boxes)
            )
            n += 1
            total_boxes += len(boxes)
            if boxes:
                with_box += 1
            if (i + 1) % 200 == 0:
                print(f"  {split}: {i + 1}/{len(images)}")

        stats[split] = dict(images=n, with_coral=with_box, boxes=total_boxes,
                            mean_boxes=round(total_boxes / max(n, 1), 2))
        print(f"{split}: {stats[split]}")

    (OUT_DIR / "data.yaml").write_text(
        f"path: {OUT_DIR.as_posix()}\ntrain: images/train\nval: images/val\nnc: 1\nnames: ['coral']\n"
    )
    print(f"\nwrote {OUT_DIR / 'data.yaml'}\n{stats}")


if __name__ == "__main__":
    main()
