"""Package a small subset of archive-2 for uploading elsewhere (Lightning
AI, Colab, etc.) instead of the full 18GB folder.

Only 1,460 of archive-2's 4,506 images have algae-class points and are
therefore ever actually read by grow_algae_masks.py - the rest were
never used. This copies just those images (real copies, not symlinks -
symlinks don't survive an upload to another machine) plus the full CSV
(29MB, trivial size, not worth filtering) into archive-2-subset/, with
the same folder layout as archive-2/ itself, so it can be renamed to
archive-2/ on the destination machine and every existing script works
unchanged.
"""
import shutil
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
# archive-2 was removed (identical CSV + image-subset of archive (1)); use whichever exists.
SOURCE_DIR = next(
    (REPO_ROOT / n for n in ("archive-2", "archive (1)", "archive-2-subset") if (REPO_ROOT / n).exists()),
    REPO_ROOT / "archive (1)",
)
CSV_PATH = SOURCE_DIR / "combined_annotations_remapped.csv"
IMAGE_DIR = SOURCE_DIR / "images" / "images"
OUT_DIR = REPO_ROOT / "archive-2-subset"
ALGAE_LABELS = {"macroalgae", "algae", "green_fleshy_algae"}


def find_image_file(name: str) -> Path | None:
    p = IMAGE_DIR / name
    if p.exists():
        return p
    stem = Path(name).stem
    matches = list(IMAGE_DIR.glob(f"{stem}.*"))
    return matches[0] if matches else None


def main():
    df = pd.read_csv(CSV_PATH)
    algae_names = sorted(df[df["Label"].isin(ALGAE_LABELS)]["Name"].unique())
    print(f"{len(algae_names)} images have algae-class points - copying just those")

    out_image_dir = OUT_DIR / "images" / "images"
    out_image_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    for name in algae_names:
        src = find_image_file(name)
        if src is None:
            print(f"WARN: {name} not found, skipping")
            continue
        shutil.copy2(src, out_image_dir / src.name)
        copied += 1

    shutil.copy2(CSV_PATH, OUT_DIR / CSV_PATH.name)
    print(f"Copied {copied} images + the annotations CSV to {OUT_DIR}")
    print("On the destination machine: rename this folder to archive-2/ and everything works unchanged.")


if __name__ == "__main__":
    main()
