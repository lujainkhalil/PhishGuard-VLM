"""
Resize and pad screenshots to a fixed square for VLM training; RGB uint8 on disk.
"""

import logging
from pathlib import Path

from PIL import Image

logger = logging.getLogger(__name__)


def resize_pad_rgb(
    image: Image.Image,
    size: int,
    *,
    background: tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    """
    Scale ``image`` to fit inside ``(size, size)``, center-pad to a square RGB image.

    Same geometry as :func:`prepare_image`; useful for in-memory inference without writing PNGs.
    """
    img = image.convert("RGB")
    w, h = img.size
    if w <= 0 or h <= 0:
        raise ValueError("Invalid image dimensions")
    scale = min(size / w, size / h)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    resized = img.resize((nw, nh), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), background)
    left = (size - nw) // 2
    top = (size - nh) // 2
    canvas.paste(resized, (left, top))
    return canvas


def prepare_image(
    src: str | Path,
    dst: str | Path,
    size: int,
    *,
    background: tuple[int, int, int] = (255, 255, 255),
) -> None:
    """
    Load image, convert to RGB, scale to fit inside (size, size), center-pad to square, save PNG.

    Pixel values remain 0–255; per-channel mean/std normalization is left to the training processor.
    """
    src, dst = Path(src), Path(dst)
    if not src.is_file():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(src) as img:
        canvas = resize_pad_rgb(img, size, background=background)
    canvas.save(dst, format="PNG")
