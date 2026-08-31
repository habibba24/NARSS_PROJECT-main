"""Grow archive-2's sparse algae point-labels into weak segmentation masks.

archive-2/combined_annotations_remapped.csv has one row per labeled point:
Name, Row, Column, Label (Row = y, Column = x, verified against image
dimensions). No boxes/masks exist in the source data, so for each point
labeled as an algae class we flood-fill a small neighborhood around it
(bounded by a radius, so it can't leak across the whole image) to get an
approximate local blob, then union all blobs for an image into one mask.

Class choice: only true algae-overgrowth types (macroalgae, algae,
green_fleshy_algae) are treated as the damage-relevant "algae" class.
crustose_coralline_algae and turf are deliberately excluded — CCA is
generally a natural/beneficial reef component, not a stress indicator,
so including it would mislabel large amounts of healthy substrate as
"damage".

Outputs, for each processed image:
- data/processed/algae_masks/masks/<stem>.png       (binary mask, for pixel IoU/Dice eval)
- data/processed/algae_masks/labels/<split>/<stem>.txt  (YOLO-seg polygon format, for training)
- data/processed/algae_masks/images/<split>/<stem>.jpg  (color-corrected copy of the source image)

Saved training images get the same gray-world color correction every
inference path applies (CoralDamagePipeline, the exported .onnx graphs) -
without it the model trains on raw underwater color casts but is always
evaluated/deployed on corrected ones, a real train/inference mismatch.
This is also why images are no longer hardlinked to archive-2's originals:
a hardlink is the same bytes as the source, so "correcting" it in place
would have corrupted archive-2 itself.
"""
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from pipeline.preprocessing import ColorCastCorrection

REPO_ROOT = Path(__file__).resolve().parents[2]
# archive-2 was a strict image-subset of archive (1) with a byte-identical CSV and
# has since been removed; fall through to whichever point-annotation folder exists.
ARCHIVE_DIR = next(
    (REPO_ROOT / n for n in ("archive (1)", "archive-2", "archive-2-subset") if (REPO_ROOT / n).exists()),
    REPO_ROOT / "archive (1)",
)
CSV_PATH = ARCHIVE_DIR / "combined_annotations_remapped.csv"
IMAGE_DIR = ARCHIVE_DIR / "images" / "images"
OUT_DIR = REPO_ROOT / "data" / "processed" / "algae_masks"
ALGAE_LABELS = {"macroalgae", "algae", "green_fleshy_algae"}
VAL_FRACTION = 0.2
SEED = 42
FLOOD_TOLERANCE = 10
MIN_CONTOUR_AREA = 20
# archive-2 photos are ~1960x1960, but YOLO trains at 640x640 - a fixed-tolerance
# color flood fill on a busy reef photo terminates almost immediately (median
# real blob was ~12px diameter at full res), which shrinks to ~4px after the
# mandatory resize to training size: literally sub-pixel at YOLO's own internal
# mask resolution (mask_ratio=4 -> 160x160 proto-masks for a 640 input), so the
# model had no learnable signal regardless of epoch count. This guarantees every
# point contributes a mask blob large enough to survive that resize.
TRAIN_IMGSZ = 640
MIN_DIAMETER_AT_TRAIN_RES = 20  # target px diameter once resized to TRAIN_IMGSZ


def find_image_file(name: str) -> Path | None:
    p = IMAGE_DIR / name
    if p.exists():
        return p
    stem = Path(name).stem
    matches = list(IMAGE_DIR.glob(f"{stem}.*"))
    return matches[0] if matches else None


def grow_mask(image_bgr: np.ndarray, points_xy: list[tuple[int, int]]) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    radius = int(np.clip(min(w, h) * 0.03, 15, 60))
    full_mask = np.zeros((h, w), dtype=np.uint8)

    # radius a point's mask must reach, at THIS image's resolution, so it's still
    # >= MIN_DIAMETER_AT_TRAIN_RES px wide after YOLO resizes down to TRAIN_IMGSZ
    scale_to_train = TRAIN_IMGSZ / max(h, w)
    min_radius = int(np.ceil((MIN_DIAMETER_AT_TRAIN_RES / 2) / scale_to_train))
    min_area = np.pi * min_radius**2

    for x, y in points_xy:
        x0, x1 = max(0, x - radius), min(w, x + radius)
        y0, y1 = max(0, y - radius), min(h, y + radius)
        if x1 <= x0 or y1 <= y0:
            continue
        roi = image_bgr[y0:y1, x0:x1].copy()
        seed = (int(np.clip(x - x0, 0, roi.shape[1] - 1)), int(np.clip(y - y0, 0, roi.shape[0] - 1)))
        ff_mask = np.zeros((roi.shape[0] + 2, roi.shape[1] + 2), dtype=np.uint8)
        try:
            cv2.floodFill(
                roi, ff_mask, seed, newVal=(255, 255, 255),
                loDiff=(FLOOD_TOLERANCE,) * 3, upDiff=(FLOOD_TOLERANCE,) * 3,
                flags=8 | cv2.FLOODFILL_FIXED_RANGE,
            )
        except cv2.error:
            continue
        blob = ff_mask[1:-1, 1:-1]

        # a busy reef photo's color flood fill often terminates on a texture edge
        # well before the radius cap, leaving a blob too small to survive the
        # later resize to training resolution - fall back to a guaranteed-size
        # disk at the labeled point so every point still contributes a learnable
        # object, instead of only ever shrinking to a noise-level speck.
        if int((blob > 0).sum()) < min_area:
            cv2.circle(blob, seed, min_radius, 1, thickness=-1)

        full_mask[y0:y1, x0:x1] = np.maximum(full_mask[y0:y1, x0:x1], blob * 255)

    return full_mask


def correct_bgr(image_bgr: np.ndarray, color_correction: ColorCastCorrection) -> np.ndarray:
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float()
    corrected = color_correction(tensor).squeeze(0).permute(1, 2, 0).clamp(0, 255).byte().numpy()
    return cv2.cvtColor(corrected, cv2.COLOR_RGB2BGR)


def mask_to_yolo_seg_lines(mask: np.ndarray, class_idx: int = 0) -> list[str]:
    h, w = mask.shape
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lines = []
    for contour in contours:
        if cv2.contourArea(contour) < MIN_CONTOUR_AREA:
            continue
        pts = contour.reshape(-1, 2).astype(np.float64)
        pts[:, 0] /= w
        pts[:, 1] /= h
        coord_str = " ".join(f"{v:.6f}" for v in pts.flatten())
        lines.append(f"{class_idx} {coord_str}")
    return lines


def main():
    random.seed(SEED)
    color_correction = ColorCastCorrection()
    df = pd.read_csv(CSV_PATH)
    algae_df = df[df["Label"].isin(ALGAE_LABELS)]
    image_names = sorted(algae_df["Name"].unique())
    print(f"{len(image_names)} images have algae-class points ({len(algae_df)} points total)")

    random.shuffle(image_names)
    n_val = max(1, int(len(image_names) * VAL_FRACTION))
    splits = {"val": image_names[:n_val], "train": image_names[n_val:]}

    (OUT_DIR / "masks").mkdir(parents=True, exist_ok=True)
    written = 0
    for split, names in splits.items():
        (OUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)

        for name in names:
            image_path = find_image_file(name)
            if image_path is None:
                print(f"WARN: image not found for {name}, skipping")
                continue
            image_bgr = cv2.imread(str(image_path))
            if image_bgr is None:
                print(f"WARN: failed to read {image_path}, skipping")
                continue
            image_bgr = correct_bgr(image_bgr, color_correction)

            points = algae_df[algae_df["Name"] == name][["Column", "Row"]].to_records(index=False)
            points_xy = [(int(c), int(r)) for c, r in points]

            mask = grow_mask(image_bgr, points_xy)
            stem = image_path.stem
            cv2.imwrite(str(OUT_DIR / "masks" / f"{stem}.png"), mask)

            lines = mask_to_yolo_seg_lines(mask)
            (OUT_DIR / "labels" / split / f"{stem}.txt").write_text("\n".join(lines) + "\n" if lines else "")
            out_image_path = OUT_DIR / "images" / split / image_path.name
            if not out_image_path.exists():
                cv2.imwrite(str(out_image_path), image_bgr)
            written += 1

        print(f"{split}: processed {len(names)} images")

    data_yaml = OUT_DIR / "data.yaml"
    data_yaml.write_text(
        f"path: {OUT_DIR.as_posix()}\ntrain: images/train\nval: images/val\nnc: 1\nnames: ['algae']\n"
    )
    print(f"Wrote {written} masks total, and {data_yaml}")


if __name__ == "__main__":
    main()
