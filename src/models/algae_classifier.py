"""Stage 2a: algae-overgrowth tile classifier.

Binary classifier ("algae" vs "not_algae") over fixed-size reef tiles
(see build_algae_tiles.py). Replaces algae_segmenter.py: CoralNet point
labels support "is there algae overgrowth in this region", not object
outlines, so a per-tile call is what the data actually gives - the weak
mask segmenter stalled at mask mAP50 ~= 0.01.

At inference (see CoralDamagePipeline._check_algae) a detected colony
crop is tiled the same way, each tile classified, and the fraction of
"algae" tiles becomes the colony's algae-coverage score + a coarse
heatmap. The tidy CNN also traces to ONNX far more cleanly than the
seg head did.
"""
from pathlib import Path

import numpy as np
from PIL import Image
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "processed" / "algae_tiles"
CHECKPOINT_DIR = REPO_ROOT / "models" / "checkpoints" / "algae_classifier"
BASE_MODEL = "yolo11n-cls.pt"

# must match build_algae_tiles.py
TILE = 224


def train(epochs: int = 30, imgsz: int = 224, device: str | None = None):
    model = YOLO(BASE_MODEL)
    results = model.train(
        data=str(DATA_DIR),
        epochs=epochs,
        imgsz=imgsz,
        device=device,
        project=str(CHECKPOINT_DIR.parent),
        name=CHECKPOINT_DIR.name,
        exist_ok=True,
        patience=10,
        # 15 GB RAM box: keep dataloader workers modest so the Windows commit
        # limit isn't blown (the seg run died on "shared file mapping" / err 1455).
        workers=4,
        batch=64,
    )
    return results


def load_best() -> YOLO:
    best_path = CHECKPOINT_DIR / "weights" / "best.pt"
    if not best_path.exists():
        raise FileNotFoundError(f"No trained checkpoint at {best_path} - run train() first")
    return YOLO(str(best_path))


def _tile_boxes(w: int, h: int, stride: int):
    """Top-left (x, y) of every TILE x TILE window over a w x h crop, last row/col
    snapped to the edge. If the crop is smaller than a tile in either axis, a
    single full-crop window is used (score_crop resizes it up)."""
    if w < TILE or h < TILE:
        yield 0, 0, w, h
        return
    xs = list(range(0, w - TILE + 1, stride)) or [0]
    ys = list(range(0, h - TILE + 1, stride)) or [0]
    if xs[-1] != w - TILE:
        xs.append(w - TILE)
    if ys[-1] != h - TILE:
        ys.append(h - TILE)
    for y in ys:
        for x in xs:
            yield x, y, TILE, TILE


def score_crop(model: YOLO, crop_rgb, stride: int = TILE, conf_thresh: float = 0.5) -> dict:
    """Tile a colony crop (uint8 RGB, already colour-corrected) the same way
    build_algae_tiles.py did, classify every tile, and summarise:

      coverage       fraction of tiles predicted 'algae' (>= conf_thresh)
      mean_prob      mean P(algae) over tiles
      n_tiles        tiles evaluated
      n_algae_tiles  tiles over threshold
      heatmap        crop-relative HxW float32, per-pixel mean P(algae)
    """
    crop_rgb = np.asarray(crop_rgb)
    if crop_rgb.ndim != 3 or crop_rgb.size == 0:
        return {"coverage": 0.0, "mean_prob": 0.0, "n_tiles": 0, "n_algae_tiles": 0,
                "heatmap": np.zeros((1, 1), np.float32)}

    h, w = crop_rgb.shape[:2]
    boxes = list(_tile_boxes(w, h, stride))
    imgs = []
    for x, y, tw, th in boxes:
        sub = crop_rgb[y : y + th, x : x + tw]
        if (tw, th) != (TILE, TILE):
            sub = np.asarray(Image.fromarray(sub).resize((TILE, TILE), Image.BILINEAR))
        imgs.append(Image.fromarray(sub))

    results = model.predict(imgs, verbose=False)
    algae_idx = next(i for i, n in model.names.items() if n == "algae")

    heat = np.zeros((h, w), np.float32)
    cnt = np.zeros((h, w), np.float32)
    probs, n_algae = [], 0
    for (x, y, tw, th), r in zip(boxes, results):
        p = float(r.probs.data[algae_idx])
        probs.append(p)
        if p >= conf_thresh:
            n_algae += 1
        heat[y : y + th, x : x + tw] += p
        cnt[y : y + th, x : x + tw] += 1.0
    cnt[cnt == 0] = 1.0

    return {
        "coverage": n_algae / len(boxes),
        "mean_prob": float(np.mean(probs)),
        "n_tiles": len(boxes),
        "n_algae_tiles": n_algae,
        "heatmap": heat / cnt,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()
    train(epochs=args.epochs, imgsz=args.imgsz, device=args.device)
