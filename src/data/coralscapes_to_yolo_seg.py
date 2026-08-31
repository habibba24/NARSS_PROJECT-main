"""Build the algae segmentation dataset from Coralscapes' real expert masks.

Replaces the weak flood-fill masks (grow_algae_masks.py) - CoralNet point
labels could never give real algae outlines, so the segmenter stalled at
mask mAP50 ~= 0.01. The Coralscapes dataset (Sauder et al. 2025,
Apache-2.0, 1517 train / 166 val Red Sea reef images) has 174k hand-drawn
polygons over 39 benthic classes; class 10 "algae covered substrate"
(turf + macroalgae + fleshy algae + the Turbinaria macroalga) is exactly
this pipeline's algae-overgrowth target.

Reads the Zenodo / Cityscapes-style layout:
    <root>/leftImg8bit/<split>/<site>/<id>_leftImg8bit.png   RGB image
    <root>/gtFine/<split>/<site>/<id>_gtFine.png             L mask, pixel = class id (0 = ignore)
    <root>/classes.json                                      {name: id}
Point CORALSCAPES_ROOT below (or the env var of the same name) at the
folder that contains leftImg8bit/ and gtFine/.

For each image:
    data/processed/algae_seg/images/<split>/<id>.jpg   raw (see note below)
    data/processed/algae_seg/labels/<split>/<id>.txt   YOLO-seg polygons, class 0 = algae
    data/processed/algae_seg/masks/<split>/<id>.png    binary mask (pixel eval)
    data/processed/algae_seg/data.yaml

Images are saved RAW, not gray-world colour-corrected. The active stage 2a
model (Coralscapes DINOv3, models/algae_segmenter.py) was trained on raw
underwater photos and loses recall on WB-corrected input, and evaluate_all
scores it against these images. The pipeline passes it raw colony crops
for the same reason. (The earlier YOLO-seg attempt on these polygons,
which did want corrected images, is abandoned - mask mAP50 ~= 0.03.)
Coralscapes' own train/val split is kept; test is ignored. Images with no
algae pixels get an empty label file (valid negative for a segmenter).

Note: point-based grow_algae_masks.py excluded turf (CCA/turf treated as
benign); Coralscapes folds turf into class 10 and has no separate turf or
CCA class, so some turf-covered substrate is included. Turf overgrowth on
substrate is itself a degradation signal - this is the one definitional
difference from the old approach.
"""
import json
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "data" / "processed" / "algae_seg"

_ROOT_CANDIDATES = [
    os.environ.get("CORALSCAPES_ROOT", ""),
    r"D:/coralscapes/coralscapes",
    r"D:/coralscapes",
    str(REPO_ROOT / "data" / "external" / "coralscapes"),
]
CORALSCAPES_ROOT = next(
    (Path(p) for p in _ROOT_CANDIDATES if p and (Path(p) / "leftImg8bit").is_dir()),
    None,
)

ALGAE_CLASS_ID = 10
MIN_CONTOUR_AREA = 80        # px^2 at native (2048x1024) res
POLY_EPS_FRAC = 0.004        # approxPolyDP epsilon as fraction of contour perimeter
JPEG_QUALITY = 90
SPLITS = ("train", "val")


def mask_to_yolo_seg_lines(binary: np.ndarray, cls: int = 0) -> list[str]:
    h, w = binary.shape
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lines = []
    for c in contours:
        if cv2.contourArea(c) < MIN_CONTOUR_AREA:
            continue
        eps = POLY_EPS_FRAC * cv2.arcLength(c, True)
        c = cv2.approxPolyDP(c, eps, True).reshape(-1, 2).astype(np.float64)
        if len(c) < 3:
            continue
        c[:, 0] /= w
        c[:, 1] /= h
        c = np.clip(c, 0.0, 1.0)
        lines.append(f"{cls} " + " ".join(f"{v:.6f}" for v in c.flatten()))
    return lines


def main():
    if CORALSCAPES_ROOT is None:
        raise SystemExit(
            "Coralscapes not found. Set CORALSCAPES_ROOT to the folder containing "
            "leftImg8bit/ and gtFine/ (tried: " + ", ".join(c for c in _ROOT_CANDIDATES if c) + ")"
        )
    classes = json.loads((CORALSCAPES_ROOT / "classes.json").read_text())
    assert classes.get("algae covered substrate") == ALGAE_CLASS_ID, classes
    print(f"Coralscapes root: {CORALSCAPES_ROOT}  (algae class id {ALGAE_CLASS_ID})")

    stats = {}
    for split in SPLITS:
        img_dir = CORALSCAPES_ROOT / "leftImg8bit" / split
        images = sorted(img_dir.rglob("*_leftImg8bit.png"))
        out_img = OUT_DIR / "images" / split
        out_lbl = OUT_DIR / "labels" / split
        out_msk = OUT_DIR / "masks" / split
        for d in (out_img, out_lbl, out_msk):
            d.mkdir(parents=True, exist_ok=True)

        n = with_algae = polys = 0
        frac_sum = 0.0
        for i, ip in enumerate(images):
            # .../leftImg8bit/<split>/<site>/<id>_leftImg8bit.png -> .../gtFine/<split>/<site>/<id>_gtFine.png
            mp = ip.parents[3] / "gtFine" / split / ip.parent.name / ip.name.replace("_leftImg8bit.png", "_gtFine.png")
            if not mp.exists():
                print(f"WARN: no mask for {ip.name}, skipping")
                continue
            stem = ip.name.replace("_leftImg8bit.png", "")

            rgb = np.array(Image.open(ip).convert("RGB"))
            lab = np.array(Image.open(mp))
            if lab.shape != rgb.shape[:2]:
                lab = np.array(Image.fromarray(lab).resize((rgb.shape[1], rgb.shape[0]), Image.NEAREST))
            binary = (lab == ALGAE_CLASS_ID).astype(np.uint8)

            Image.fromarray(rgb).save(out_img / f"{stem}.jpg", "JPEG", quality=JPEG_QUALITY)
            cv2.imwrite(str(out_msk / f"{stem}.png"), binary * 255)
            lines = mask_to_yolo_seg_lines(binary) if binary.any() else []
            (out_lbl / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))

            n += 1
            polys += len(lines)
            if binary.any():
                with_algae += 1
                frac_sum += float(binary.mean())
            if (i + 1) % 200 == 0:
                print(f"  {split}: {i + 1}/{len(images)}")

        stats[split] = dict(images=n, with_algae=with_algae, polygons=polys,
                            mean_algae_frac=round(frac_sum / max(with_algae, 1), 4))
        print(f"{split}: {stats[split]}")

    (OUT_DIR / "data.yaml").write_text(
        f"path: {OUT_DIR.as_posix()}\ntrain: images/train\nval: images/val\nnc: 1\nnames: ['algae']\n"
    )
    print(f"\nwrote {OUT_DIR / 'data.yaml'}")
    print(stats)


if __name__ == "__main__":
    main()
