"""Stage 2b: bleaching detector - deterministic color math, not trained.

Per the project decision: bleaching has a real, measurable visual signal
(loss of pigment -> whitening), so instead of needing labeled examples
this scores each pixel by how far it has moved from ITS OWN coral
genus's normal color toward white - not by raw "distance from white" -
so naturally pale species (e.g. WavingHand) don't get falsely flagged,
and species-specific paling is caught even before a colony looks
absolute-white.

Pure torch tensor ops (no OpenCV) so this traces/exports to ONNX
directly alongside the rest of the pipeline. Lab conversion reproduces
OpenCV's 8-bit-scaled COLOR_BGR2LAB convention on purpose, to match the
genus reference stats computed offline via cv2 in prep_bleaching_reference.py.
"""
import json
from pathlib import Path

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[2]
REFERENCE_PATH = REPO_ROOT / "data" / "processed" / "bleaching_reference" / "genus_colors.json"
WHITE_LAB_8U = torch.tensor([255.0, 128.0, 128.0])


def rgb_to_lab_8u(rgb: torch.Tensor) -> torch.Tensor:
    """rgb: [B,3,H,W] in [0,255]. Returns Lab in OpenCV's 8-bit-scaled convention."""
    srgb = (rgb / 255.0).clamp(0.0, 1.0)
    linear = torch.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4)
    r, g, b = linear[:, 0], linear[:, 1], linear[:, 2]

    x = 0.4124564 * r + 0.3575761 * g + 0.1804375 * b
    y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b
    z = 0.0193339 * r + 0.1191920 * g + 0.9503041 * b

    xn, yn, zn = 0.95047, 1.0, 1.08883
    delta = 6.0 / 29.0

    def f(t):
        return torch.where(t > delta**3, t.clamp(min=1e-8) ** (1.0 / 3.0), t / (3 * delta**2) + 4.0 / 29.0)

    fx, fy, fz = f(x / xn), f(y / yn), f(z / zn)
    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b_ = 200.0 * (fy - fz)

    lab = torch.stack([L * 255.0 / 100.0, a + 128.0, b_ + 128.0], dim=1)
    return lab.clamp(0.0, 255.0)


class BleachingModule(nn.Module):
    """forward(rgb, genus_idx) -> paling_score [0,1] and binary mask per pixel.

    genus_idx comes from the colony detector's predicted class for each
    crop; there's no "unknown genus" fallback in v1 - a crop must map to
    one of coral_soft's 6 reference genera.
    """

    def __init__(self, genus_names: list[str] | None = None, threshold: float = 0.3):
        # 0.3 calibrated against the 923-image Kaggle healthy/bleached set via
        # calibrate_bleaching_threshold.py (precision=0.600, recall=0.482, f1=0.535
        # at this value paired with CoralDamagePipeline's bleaching_fraction_cutoff=0.0) -
        # not a guess, see data/processed/bleaching_reference/threshold_calibration.csv
        super().__init__()
        reference = json.loads(REFERENCE_PATH.read_text()) if REFERENCE_PATH.exists() else {}
        names = genus_names or sorted(reference.keys())
        if not names:
            raise ValueError(f"No genus color reference at {REFERENCE_PATH} - run prep_bleaching_reference.py first")
        self.genus_names = names
        mean_lab = torch.tensor([reference[n]["mean_lab"] for n in names], dtype=torch.float32)
        self.register_buffer("genus_mean_lab", mean_lab)
        self.register_buffer("white_lab", WHITE_LAB_8U.clone())
        self.threshold = threshold

    def forward(self, rgb: torch.Tensor, genus_idx: torch.Tensor):
        lab = rgb_to_lab_8u(rgb)  # [B,3,H,W]
        mean = self.genus_mean_lab[genus_idx]  # [B,3]
        white = self.white_lab.unsqueeze(0).expand_as(mean)  # [B,3]

        direction = white - mean  # [B,3] - the "paling direction" for each crop's genus
        denom = (direction * direction).sum(dim=1).clamp(min=1e-6)

        pixel = lab.permute(0, 2, 3, 1)  # [B,H,W,3]
        delta = pixel - mean.view(-1, 1, 1, 3)
        proj = (delta * direction.view(-1, 1, 1, 3)).sum(dim=-1) / denom.view(-1, 1, 1)
        score = proj.clamp(0.0, 1.0)

        return {"paling_score": score, "mask": score > self.threshold}
