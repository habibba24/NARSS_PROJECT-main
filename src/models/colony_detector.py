"""Stage 1: coral colony detector (AQUARIUM path only).

Fine-tunes a YOLO detector on coral_soft's genus-labeled boxes to localize
"this region is coral tissue" so the downstream per-colony bleaching check
looks inside real coral, not sand/background/glare. All 6 genera are folded
into localizing coral-vs-not.

Scope note: this only works on aquarium-style close-ups (its training
domain). On wide reef photos it finds nothing - retraining it on
Coralscapes-derived boxes (coralscapes_to_yolo_det.py) only reached
mAP50 ~0.34, because reef "coral" is a semantic region, not discrete
objects. Reef photos take a different route entirely -
CoralDamagePipeline.run_reef(), which reads coral / bleached / algae off
one Coralscapes segmentation pass and uses no colony detector at all.
"""
from pathlib import Path

from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_YAML = REPO_ROOT / "data" / "processed" / "coral_soft" / "data.yaml"
CHECKPOINT_DIR = REPO_ROOT / "models" / "checkpoints" / "colony_detector"
BASE_MODEL = "yolo11n.pt"


def train(epochs: int = 30, imgsz: int = 640, device: str | None = None):
    model = YOLO(BASE_MODEL)
    results = model.train(
        data=str(DATA_YAML),
        epochs=epochs,
        imgsz=imgsz,
        device=device,
        project=str(CHECKPOINT_DIR.parent),
        name=CHECKPOINT_DIR.name,
        exist_ok=True,
        patience=10,
        # 15 GB RAM Windows box: no image cache + few workers is the stable
        # combination (cache='ram' + spawned workers truncates their pickle).
        workers=2,
        batch=16,
        cache=False,
    )
    return results


def load_best() -> YOLO:
    best_path = CHECKPOINT_DIR / "weights" / "best.pt"
    if not best_path.exists():
        raise FileNotFoundError(f"No trained checkpoint at {best_path} - run train() first")
    return YOLO(str(best_path))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()
    train(epochs=args.epochs, imgsz=args.imgsz, device=args.device)
