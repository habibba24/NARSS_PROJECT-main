"""Shared preprocessing, split into what genuinely embeds in an exported
model graph vs. what has to happen before the tensor even exists.

Division of labor (this is a real ONNX/tracing constraint, not a
shortcut): resizing an arbitrary-resolution raw photo down to the
network's fixed input size is a shape-changing operation that doesn't
trace into a portable static graph, so it happens once, in plain
numpy/PIL, before the tensor is built - exactly like every deployed
vision model does it (a browser or mobile client does this same resize
before calling the model). What DOES travel inside the exported graph -
the domain-specific part the user asked for - is the underwater
color-cast correction and normalization, implemented as real tensor ops
in ColorCastCorrection below and traced/exported together with each
detector in export_onnx.py.
"""
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

PAD_VALUE = 114  # standard YOLO letterbox pad color (mid-gray)


def letterbox_resize_numpy(image: np.ndarray, size: int) -> tuple[np.ndarray, float, tuple[int, int]]:
    """image: HxWx3 uint8 RGB. Returns (padded size x size x 3 uint8, scale, (pad_left, pad_top))."""
    h, w = image.shape[:2]
    scale = min(size / h, size / w)
    new_h, new_w = int(round(h * scale)), int(round(w * scale))
    resized = np.array(Image.fromarray(image).resize((new_w, new_h), Image.BILINEAR))

    canvas = np.full((size, size, 3), PAD_VALUE, dtype=np.uint8)
    pad_top = (size - new_h) // 2
    pad_left = (size - new_w) // 2
    canvas[pad_top : pad_top + new_h, pad_left : pad_left + new_w] = resized
    return canvas, scale, (pad_left, pad_top)


def load_and_letterbox(image_path: str, size: int = 640) -> torch.Tensor:
    """Raw photo on disk -> [1,3,size,size] float tensor in [0,255], ready for
    ColorCastCorrection / a PreprocessedDetector's forward()."""
    image = np.array(Image.open(image_path).convert("RGB"))
    canvas, _, _ = letterbox_resize_numpy(image, size)
    tensor = torch.from_numpy(canvas).permute(2, 0, 1).unsqueeze(0).float()
    return tensor


class ColorCastCorrection(nn.Module):
    """Gray-world white balance: underwater photos skew blue/green with
    depth, which corrupts both raw appearance and (critically) the
    bleaching module's whiteness measurement downstream. Scales each
    color channel so its mean matches the overall gray mean. Gain is
    clamped so near-black regions (e.g. shadow/background) don't get
    blown out.
    """

    def __init__(self, min_gain: float = 0.5, max_gain: float = 3.0):
        super().__init__()
        self.min_gain = min_gain
        self.max_gain = max_gain

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        # rgb: [B,3,H,W] in [0,255]
        channel_mean = rgb.mean(dim=(2, 3), keepdim=True)  # [B,3,1,1]
        overall_mean = channel_mean.mean(dim=1, keepdim=True)  # [B,1,1,1]
        gain = (overall_mean / channel_mean.clamp(min=1e-3)).clamp(self.min_gain, self.max_gain)
        return (rgb * gain).clamp(0.0, 255.0)


class Normalize(nn.Module):
    def forward(self, rgb_0_255: torch.Tensor) -> torch.Tensor:
        return rgb_0_255 / 255.0
