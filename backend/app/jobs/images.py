"""Validation and normalisation of uploaded I2V input images."""

from __future__ import annotations

import io
import warnings
from dataclasses import dataclass

from PIL import Image, ImageOps, UnidentifiedImageError

ALLOWED_FORMATS = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp"}
# Pillow raises DecompressionBombError above 2x this. Guards against tiny files that
# decode into huge images.
Image.MAX_IMAGE_PIXELS = 40_000_000


class ImageValidationError(ValueError):
    def __init__(self, message: str, status_code: int = 422):
        super().__init__(message)
        self.status_code = status_code


@dataclass(frozen=True)
class ValidatedImage:
    png_bytes: bytes          # normalised: RGB PNG, EXIF orientation applied, metadata stripped
    width: int
    height: int
    original_format: str
    original_size_bytes: int

    def info(self) -> dict:
        return {
            "width": self.width, "height": self.height,
            "original_format": self.original_format.lower(),
            "original_size_bytes": self.original_size_bytes,
        }


def validate_image(data: bytes, *, max_bytes: int, min_side: int, max_side: int) -> ValidatedImage:
    if not data:
        raise ImageValidationError("image file is empty")
    if len(data) > max_bytes:
        raise ImageValidationError(
            f"image is too large ({len(data)} bytes); limit is {max_bytes} bytes", status_code=413)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as probe:
                fmt = probe.format
                probe.verify()  # structural check without full decode
            if fmt not in ALLOWED_FORMATS:
                raise ImageValidationError(
                    f"unsupported image format {fmt!r}; allowed: JPEG, PNG, WEBP", status_code=415)
            with Image.open(io.BytesIO(data)) as img:
                img.load()  # full decode, catches truncated files
                img = ImageOps.exif_transpose(img)
                if getattr(img, "n_frames", 1) > 1:
                    img.seek(0)
                rgb = img.convert("RGB")
    except ImageValidationError:
        raise
    except (Image.DecompressionBombError, Image.DecompressionBombWarning):
        raise ImageValidationError("image dimensions are too large") from None
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError):
        raise ImageValidationError("file is not a valid image") from None

    w, h = rgb.size
    if min(w, h) < min_side:
        raise ImageValidationError(f"image is too small ({w}x{h}); shortest side must be >= {min_side}px")
    if max(w, h) > max_side:
        raise ImageValidationError(f"image is too large ({w}x{h}); longest side must be <= {max_side}px")

    buf = io.BytesIO()
    rgb.save(buf, format="PNG", optimize=False)
    return ValidatedImage(buf.getvalue(), w, h, fmt, len(data))
