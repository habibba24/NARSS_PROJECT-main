"""The unified pipeline.

Two different things live here, deliberately not the same thing:

1. `PreprocessedDetector` - an nn.Module used only by export_onnx.py. It
   chains ColorCastCorrection + Normalize + one trained YOLO stage's raw
   torch model into a single traceable graph, so each exported .onnx
   file for the website/app genuinely does its own color correction
   internally (the user's explicit requirement).

2. `CoralDamagePipeline` - a plain Python class used for local
   development/eval/demo in this repo. It runs the full colony -> algae
   / bleaching sequence using each stage's normal ultralytics `.predict()`
   (which already handles its own resize + NMS), since that Python-level
   control flow (crop to each detected box, feed the crop to downstream
   stages) is not something that reasonably traces into one static ONNX
   graph - see the README for why each stage is exported separately with
   a thin orchestration layer at the app layer instead of one monolithic
   exported graph.

Note: a disease detector was originally planned as a third damage type,
fine-tuned on an external Roboflow dataset. That dataset turned out to
only have genus/species labels, not disease labels, and no other source
was available, so disease detection was dropped from scope - this
pipeline covers algae and bleaching only.
"""
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from models import colony_detector, algae_segmenter
from models.bleaching_module import BleachingModule
from pipeline.preprocessing import ColorCastCorrection, Normalize


class PreprocessedDetector(nn.Module):
    """color-correction + normalize + a trained YOLO stage's torch model,
    as one exportable nn.Module. `raw_model` is expected to be the
    underlying `ultralytics.YOLO(...).model` (a plain nn.Module), not the
    high-level YOLO wrapper (which does Python-side NMS/decoding that
    doesn't trace cleanly).
    """

    def __init__(self, raw_model: nn.Module):
        super().__init__()
        self.color_correction = ColorCastCorrection()
        self.normalize = Normalize()
        self.raw_model = raw_model

    def forward(self, rgb_0_255: torch.Tensor):
        corrected = self.color_correction(rgb_0_255)
        normalized = self.normalize(corrected)
        return self.raw_model(normalized)


@dataclass
class DamageFinding:
    damage_type: str  # "algae" | "bleaching"
    colony_box_xyxy: tuple[float, float, float, float]  # in original image coords
    colony_genus: str
    confidence: float
    detail: dict = field(default_factory=dict)


class CoralDamagePipeline:
    def __init__(
        self,
        device: str | None = None,
        bleaching_fraction_cutoff: float = 0.0,
        algae_coverage_cutoff: float = 0.0,
    ):
        self.colony_model = colony_detector.load_best()
        self.algae_model = algae_segmenter.load_model(device)
        self.bleaching_module = BleachingModule()
        self.color_correction = ColorCastCorrection()
        self.device = device
        # fraction of a colony crop's pixels the algae segmenter must mark as
        # "algae covered substrate" before the colony is flagged for overgrowth.
        # 0.0 = any algae pixel is enough (mirrors bleaching_fraction_cutoff);
        # raise it to suppress specks. Coralscapes DINOv3 segmenter, algae class
        # on Coralscapes val: IoU 0.56, recall 0.86, precision 0.62.
        self.algae_coverage_cutoff = algae_coverage_cutoff
        # what fraction of a colony's pixels must be flagged pale before the colony
        # itself counts as "bleached" - calibrated against the 923-image Kaggle
        # healthy/bleached set via calibrate_bleaching_threshold.py (paired with
        # BleachingModule's threshold=0.3): precision=0.600, recall=0.482, f1=0.535.
        # 0.0 means a single sufficiently-pale pixel is enough to flag the colony -
        # this is what the real data supported, not an arbitrary choice; see
        # data/processed/bleaching_reference/threshold_calibration.csv for the full sweep.
        self.bleaching_fraction_cutoff = bleaching_fraction_cutoff

    def _correct_image(self, image_rgb: np.ndarray) -> np.ndarray:
        tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).unsqueeze(0).float()
        corrected = self.color_correction(tensor)
        return corrected.squeeze(0).permute(1, 2, 0).clamp(0, 255).byte().numpy()

    def run(self, image_path: str, colony_conf: float = 0.1) -> list[DamageFinding]:
        # 0.25 was an arbitrary default that turned out to sit above this
        # (still lightly-trained) model's typical confidence scores, so it
        # was silently producing zero detections on nearly everything,
        # including the model's own training-distribution images. 0.1 was
        # picked by inspecting the actual confidence distribution - see
        # the diagnosis in calibrate_bleaching_threshold.py's usage notes.
        # Revisit once the detector is trained further / on more data.
        raw = np.array(Image.open(image_path).convert("RGB"))
        corrected = self._correct_image(raw)
        corrected_pil = Image.fromarray(corrected)

        findings: list[DamageFinding] = []
        colony_results = self.colony_model.predict(corrected_pil, conf=colony_conf, verbose=False)[0]
        genus_names = colony_results.names

        for box in colony_results.boxes:
            x0, y0, x1, y1 = (float(v) for v in box.xyxy[0])
            genus = genus_names[int(box.cls[0])]
            box_conf = float(box.conf[0])
            iy0, iy1, ix0, ix1 = int(y0), int(y1), int(x0), int(x1)
            crop = corrected[iy0:iy1, ix0:ix1]
            if crop.size == 0:
                continue

            # the algae segmenter (Coralscapes DINOv3) was trained on raw
            # underwater photos - gray-world correction is off-distribution for
            # it and drops recall, so it gets the raw crop. bleaching genuinely
            # needs the colour-corrected crop (it measures paleness after WB).
            raw_crop = raw[iy0:iy1, ix0:ix1]
            findings.extend(self._check_algae(raw_crop, (x0, y0, x1, y1), genus, box_conf))
            findings.extend(self._check_bleaching(crop, (x0, y0, x1, y1), genus, box_conf))

        return findings

    def run_reef(self, image_path: str) -> list[DamageFinding]:
        """Reef-photo path: one whole-image semantic pass with the Coralscapes
        segmenter gives coral / bleached-coral / algae directly - no colony
        detector (trained on aquarium close-ups, finds nothing on reef scenes)
        and no colour-based bleaching module (Coralscapes labels bleaching).
        Each returned finding is a whole-frame damage class; colony_box_xyxy is
        the full image and colony_genus is "reef".
        """
        raw = np.array(Image.open(image_path).convert("RGB"))
        h, w = raw.shape[:2]
        r = algae_segmenter.segment_reef(self.algae_model, raw)
        whole = (0.0, 0.0, float(w), float(h))
        findings: list[DamageFinding] = []

        if r["algae_cover"] > self.algae_coverage_cutoff:
            findings.append(DamageFinding(
                damage_type="algae", colony_box_xyxy=whole, colony_genus="reef",
                confidence=r["algae_cover"],
                detail={"coverage": r["algae_cover"], "mask": r["algae_mask"],
                        "heatmap": r["algae_mask"].astype("float32"),
                        "n_algae_px": r["algae_px"], "n_px": r["total_px"]},
            ))
        if r["bleached_frac"] > self.bleaching_fraction_cutoff:
            findings.append(DamageFinding(
                damage_type="bleaching", colony_box_xyxy=whole, colony_genus="reef",
                confidence=r["bleached_frac"],
                detail={"bleached_fraction_of_coral": r["bleached_frac"],
                        "frame_coverage": r["bleached_px"] / r["total_px"],
                        "mask": r["bleached_mask"], "n_bleached_px": r["bleached_px"],
                        "coral_cover": r["coral_cover"]},
            ))
        return findings

    def _check_algae(self, crop, colony_box, genus, colony_conf) -> list[DamageFinding]:
        # semantic-segment the colony crop; algae coverage + crop-relative masks.
        res = algae_segmenter.segment_crop(self.algae_model, crop)
        if res["total_px"] == 0 or res["coverage"] <= self.algae_coverage_cutoff:
            return []
        return [
            DamageFinding(
                damage_type="algae",
                colony_box_xyxy=colony_box,
                colony_genus=genus,
                confidence=res["coverage"],
                detail={
                    "coverage": res["coverage"],
                    "n_algae_px": res["algae_px"],
                    "n_px": res["total_px"],
                    # crop-relative results: bool mask (argmax) + float P(algae) map.
                    "mask": res["mask"],
                    "heatmap": res["heatmap"],
                },
            )
        ]

    def _check_bleaching(self, crop, colony_box, genus, colony_conf) -> list[DamageFinding]:
        if genus not in self.bleaching_module.genus_names:
            return []
        tensor = torch.from_numpy(crop).permute(2, 0, 1).unsqueeze(0).float()
        genus_idx = torch.tensor([self.bleaching_module.genus_names.index(genus)])
        out = self.bleaching_module(tensor, genus_idx)
        score_map = out["paling_score"][0]
        mask = out["mask"][0]
        bleached_fraction = mask.float().mean().item()
        if bleached_fraction > self.bleaching_fraction_cutoff:
            return [
                DamageFinding(
                    damage_type="bleaching",
                    colony_box_xyxy=colony_box,
                    colony_genus=genus,
                    confidence=bleached_fraction,
                    detail={
                        "mean_paling_score": float(score_map.mean()),
                        # the actual per-pixel result, crop-relative (HxW bool) -
                        # this is the segmentation output itself, not a summary of it.
                        "mask": mask.numpy(),
                        "paling_score_map": score_map.numpy(),
                    },
                )
            ]
        return []
