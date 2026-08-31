"""Stage 2a: algae-overgrowth segmenter - Coralscapes DINOv3 model.

Wraps EPFL-ECEO/coralscapes-vit-b-dpt (DINOv3 ViT-B backbone + DPT head,
Apache-2.0), a dense semantic segmenter over Coralscapes' 39 benthic
classes. Class 10, "algae covered substrate" (turf + macroalgae + fleshy
algae + the Turbinaria macroalga), is this pipeline's algae-overgrowth
target. On the Coralscapes val split: algae pixel IoU 0.56, recall 0.86,
precision 0.62, F1 0.72.

Earlier approaches, kept in the repo but not used:
  - grow_algae_masks.py + YOLO-seg on flood-filled CoralNet-point blobs
    -> mask mAP50 ~= 0.01 (points are cover samples, not outlines)
  - coralscapes_to_yolo_seg.py + YOLO-seg on the real Coralscapes polygons
    -> mask mAP50 ~= 0.03 (instance segmentation is the wrong tool for an
    amorphous region class)
  - algae_classifier.py, a 224px tile classifier on the point labels
    (top-1 0.82, algae F1 0.67) - the lightweight fallback if this ViT
    doesn't transfer to the deployment imagery.

REQUIRES: `huggingface-cli login` once (the DINOv3 backbone is a gated
Meta repo). After the first successful load everything is cached locally.

DEPLOYMENT NOTE: unlike the yolo11n stages this is a ~440 MB ViT loaded
via transformers + trust-remote-code. ONNX export for the no-Python
target is a separate task (torch.onnx on the eval module, replicating the
processor's normalisation client-side); export_onnx.py currently skips it.
"""
import importlib.util
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO_ID = "EPFL-ECEO/coralscapes-vit-b-dpt"
ALGAE_CLASS_ID = 10
PATCH_MULTIPLE = 16          # model needs H, W divisible by this
MAX_LONG_SIDE = 1024         # cap crop inference size for speed/memory
FULL_MAX_LONG_SIDE = 1536    # whole-photo inference (reef path) - a bit more detail

# Coralscapes class-id groups (see D:/coralscapes/.../classes.json) used by the
# reef path to read damage signals straight off one segmentation pass.
CORAL_ALIVE_IDS = frozenset({6, 17, 21, 22, 25, 27, 28, 31, 34, 36})
CORAL_BLEACHED_IDS = frozenset({4, 16, 19, 33})
CORAL_DEAD_IDS = frozenset({3, 20, 23, 32, 37})
CORAL_ALL_IDS = CORAL_ALIVE_IDS | CORAL_BLEACHED_IDS | CORAL_DEAD_IDS

_MODEL = None
_DEVICE = None


def _run(model, im_rgb_uint8, max_long_side):
    """Forward the model on a PIL/np RGB image, return an H0xW0 class-id map
    at the ORIGINAL resolution and the softmax probability volume (C,H,W)
    at inference resolution."""
    arr = np.asarray(im_rgb_uint8)
    h0, w0 = arr.shape[:2]
    scale = min(1.0, max_long_side / max(w0, h0))
    nw = max(PATCH_MULTIPLE, round(w0 * scale / PATCH_MULTIPLE) * PATCH_MULTIPLE)
    nh = max(PATCH_MULTIPLE, round(h0 * scale / PATCH_MULTIPLE) * PATCH_MULTIPLE)
    im = Image.fromarray(arr).resize((nw, nh), Image.BILINEAR)
    dev = _DEVICE or next(model.parameters()).device
    batch = model.processor(images=im, return_tensors="pt", do_resize=False)["pixel_values"].to(dev)
    with torch.no_grad():
        logits = model(batch)                       # [1, C, nh, nw]
    prob = logits.softmax(1)[0].float().cpu().numpy()
    pred = prob.argmax(0).astype(np.int16)
    pred_full = np.asarray(Image.fromarray(pred.astype(np.uint8)).resize((w0, h0), Image.NEAREST))
    return pred_full.astype(np.int16), prob


def load_model(device: str | None = None):
    """Download (once) + build the Coralscapes segmenter, cached per process."""
    global _MODEL, _DEVICE
    if _MODEL is not None:
        return _MODEL
    from huggingface_hub import snapshot_download

    root = Path(snapshot_download(REPO_ID))
    spec = importlib.util.spec_from_file_location("coralscapes_hub_model", root / "coralscapes_hub_model.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    _DEVICE = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
    _MODEL = mod.Dinov3DPTSegmenter.from_pretrained(root, map_location=_DEVICE).eval().to(_DEVICE)
    return _MODEL


def load_best():
    """API-compatible alias (this stage is pretrained, nothing to train)."""
    return load_model()


def segment_crop(model, crop_rgb) -> dict:
    """Run the segmenter on a colony crop (uint8 RGB) and summarise the
    algae channel:

      coverage   fraction of crop pixels predicted algae
      mask       crop-relative HxW bool (argmax == algae)
      heatmap    crop-relative HxW float32, softmax P(algae) - for display
      algae_px / total_px
    """
    crop_rgb = np.asarray(crop_rgb)
    if crop_rgb.ndim != 3 or crop_rgb.size == 0:
        z = np.zeros((1, 1), np.float32)
        return {"coverage": 0.0, "mask": z.astype(bool), "heatmap": z, "algae_px": 0, "total_px": 0}

    h, w = crop_rgb.shape[:2]
    pred, prob = _run(model, crop_rgb, MAX_LONG_SIDE)
    mask = pred == ALGAE_CLASS_ID
    prob_algae = prob[ALGAE_CLASS_ID]
    heat = np.asarray(Image.fromarray((prob_algae * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR)) / 255.0

    algae_px = int(mask.sum())
    total_px = int(mask.size)
    return {
        "coverage": algae_px / total_px if total_px else 0.0,
        "mask": mask,
        "heatmap": heat.astype(np.float32),
        "algae_px": algae_px,
        "total_px": total_px,
    }


def segment_reef(model, image_rgb) -> dict:
    """Whole-photo pass for the reef pipeline. Reads coral / bleached-coral /
    algae straight off one segmentation, no colony detector or colour module.

      pred            HxW int class-id map (original resolution)
      coral_mask, bleached_mask, dead_mask, algae_mask   HxW bool
      coral_cover     fraction of the frame that is coral (any state)
      algae_cover     fraction of the frame that is algae-covered substrate
      bleached_frac   bleached_px / (alive + bleached) coral px  -- "how much
                      of the coral is bleached", 0 if no coral in frame
      *_px            raw pixel counts
    """
    arr = np.asarray(image_rgb)
    pred, _ = _run(model, arr, FULL_MAX_LONG_SIDE)
    total = int(pred.size)

    alive = np.isin(pred, list(CORAL_ALIVE_IDS))
    bleached = np.isin(pred, list(CORAL_BLEACHED_IDS))
    dead = np.isin(pred, list(CORAL_DEAD_IDS))
    coral = alive | bleached | dead
    algae = pred == ALGAE_CLASS_ID

    coral_px = int(coral.sum())
    bleached_px = int(bleached.sum())
    algae_px = int(algae.sum())
    live_or_bleached = int(alive.sum()) + bleached_px
    return {
        "pred": pred,
        "coral_mask": coral, "bleached_mask": bleached, "dead_mask": dead, "algae_mask": algae,
        "coral_cover": coral_px / total, "algae_cover": algae_px / total,
        "bleached_frac": (bleached_px / live_or_bleached) if live_or_bleached else 0.0,
        "coral_px": coral_px, "bleached_px": bleached_px, "algae_px": algae_px, "total_px": total,
    }


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Segment algae in an image with the Coralscapes model.")
    ap.add_argument("image")
    a = ap.parse_args()
    m = load_model()
    r = segment_crop(m, np.array(Image.open(a.image).convert("RGB")))
    print(f"algae coverage: {r['coverage']:.3f}  ({r['algae_px']}/{r['total_px']} px)")
