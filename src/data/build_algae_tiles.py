"""Stage 2a data prep — algae overgrowth as a tile-classification problem.

Replaces the weak-mask segmentation approach (grow_algae_masks.py +
algae_segmenter.py), which stalled at mask mAP50 ~= 0.01: CoralNet-style
point labels are stratified *samples* of benthic cover, not object
outlines, so flood-filled blobs around them gave a segmenter no coherent
target. What the points genuinely support is "does this patch of reef
contain algae overgrowth" - a binary call on a local region.

So: color-correct each archive (1) image (same gray-world correction
every inference path applies), cut it into a grid of fixed-size tiles,
and label each tile from the points that fall inside it:

  - >= 1 point in {macroalgae, algae, green_fleshy_algae}   -> "algae"
  - >= 1 point, none of them an algae class                 -> "not_algae"
  - no points in the tile                                   -> dropped (no supervision)

crustose_coralline_algae and turf are NOT algae-overgrowth classes here
(CCA is a natural/beneficial reef component) - tiles whose only points
are CCA/turf become strong "not_algae" examples, which is correct.

Split is by source image (md5 of the name), so tiles from one photo
never land in both train and val. not_algae tiles are capped per image
at NEG_RATIO x that image's algae tiles (plus a small floor) to stop the
majority class from swamping training.

Output (Ultralytics classification layout):
  data/processed/algae_tiles/<split>/<class>/<stem>_r<row>_c<col>.jpg
  data/processed/algae_tiles/meta.json
"""
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFile

from pipeline.preprocessing import ColorCastCorrection

ImageFile.LOAD_TRUNCATED_IMAGES = True  # ~29 archive (1) JPEGs have a truncated tail

REPO_ROOT = Path(__file__).resolve().parents[2]
ARCHIVE_DIR = next(
    (REPO_ROOT / n for n in ("archive (1)", "archive-2", "archive-2-subset") if (REPO_ROOT / n).exists()),
    REPO_ROOT / "archive (1)",
)
CSV_PATH = ARCHIVE_DIR / "combined_annotations_remapped.csv"
IMAGE_DIR = ARCHIVE_DIR / "images" / "images"
OUT_DIR = REPO_ROOT / "data" / "processed" / "algae_tiles"

ALGAE_LABELS = {"macroalgae", "algae", "green_fleshy_algae"}
TILE = 224           # px; also the classifier input size, so no resize on the happy path
STRIDE = 224         # non-overlapping
MIN_TILE_FRAC = 0.6  # keep an edge tile only if it's >= this fraction of a full tile
VAL_MOD = 5          # md5(name) % VAL_MOD == 0 -> val
NEG_RATIO = 3.0      # not_algae tiles per image <= NEG_RATIO * that image's algae tiles ...
NEG_FLOOR = 2        # ... but always allow at least this many, so all-negative images still contribute
SEED = 42


def split_of(name: str) -> str:
    return "val" if int(hashlib.md5(name.encode()).hexdigest(), 16) % VAL_MOD == 0 else "train"


def find_image_file(name: str) -> Path | None:
    p = IMAGE_DIR / name
    if p.exists():
        return p
    matches = list(IMAGE_DIR.glob(f"{Path(name).stem}.*"))
    return matches[0] if matches else None


def load_points():
    """name -> list[(x, y, is_algae)] using stdlib csv (pandas not required)."""
    import csv as _csv

    pts: dict[str, list[tuple[int, int, bool]]] = defaultdict(list)
    with open(CSV_PATH, newline="", encoding="utf-8", errors="replace") as f:
        r = _csv.reader(f)
        next(r)
        for row in r:
            c = [x.strip() for x in row]
            if len(c) < 4 or not c[0] or not c[1].lstrip("-").isdigit() or not c[2].lstrip("-").isdigit():
                continue
            name, y, x, label = c[0], int(c[1]), int(c[2]), c[3]
            pts[name].append((x, y, label in ALGAE_LABELS))
    return pts


def tile_grid(w: int, h: int):
    for ty in range(0, h, STRIDE):
        for tx in range(0, w, STRIDE):
            tw, th = min(TILE, w - tx), min(TILE, h - ty)
            if tw < TILE * MIN_TILE_FRAC or th < TILE * MIN_TILE_FRAC:
                continue
            yield tx, ty, tw, th


def main():
    random.seed(SEED)
    cc = ColorCastCorrection()
    points_by_image = load_points()
    names = sorted(points_by_image)
    print(f"{len(names)} annotated images in {ARCHIVE_DIR.name}")

    for split in ("train", "val"):
        for cls in ("algae", "not_algae"):
            (OUT_DIR / split / cls).mkdir(parents=True, exist_ok=True)

    counts = Counter()
    per_split_img = Counter()
    missing = 0

    for name in names:
        img_path = find_image_file(name)
        if img_path is None:
            missing += 1
            continue
        try:
            im = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"WARN: cannot open {name}: {e}")
            missing += 1
            continue

        arr = np.asarray(im, dtype=np.float32)
        t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
        corrected = cc(t).squeeze(0).permute(1, 2, 0).clamp(0, 255).byte().numpy()
        h, w = corrected.shape[:2]
        split = split_of(name)
        per_split_img[split] += 1
        stem = img_path.stem

        pos_tiles, neg_tiles = [], []
        for tx, ty, tw, th in tile_grid(w, h):
            inside = [
                (x, y, a) for (x, y, a) in points_by_image[name]
                if tx <= x < tx + tw and ty <= y < ty + th
            ]
            if not inside:
                continue
            crop = corrected[ty : ty + th, tx : tx + tw]
            label = "algae" if any(a for *_, a in inside) else "not_algae"
            (pos_tiles if label == "algae" else neg_tiles).append((tx, ty, crop, label))

        keep_neg = max(NEG_FLOOR, int(round(NEG_RATIO * len(pos_tiles))))
        random.shuffle(neg_tiles)
        chosen = pos_tiles + neg_tiles[:keep_neg]

        for tx, ty, crop, label in chosen:
            out = OUT_DIR / split / label / f"{stem}_r{ty}_c{tx}.jpg"
            Image.fromarray(crop).save(out, "JPEG", quality=90)
            counts[(split, label)] += 1

    meta = {
        "source": str(ARCHIVE_DIR),
        "tile_px": TILE,
        "stride_px": STRIDE,
        "algae_labels": sorted(ALGAE_LABELS),
        "neg_ratio": NEG_RATIO,
        "images_used": {k: per_split_img[k] for k in ("train", "val")},
        "images_missing": missing,
        "tiles": {f"{s}/{c}": counts[(s, c)] for s in ("train", "val") for c in ("algae", "not_algae")},
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta["tiles"], indent=2))
    print(f"images missing on disk: {missing}")
    print(f"wrote tiles under {OUT_DIR}")


if __name__ == "__main__":
    main()
