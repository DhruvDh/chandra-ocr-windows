"""Prepare original data-URL images for the pinned Qwen/llama image-size identity path."""
import base64
import copy
from dataclasses import dataclass
import hashlib
import io
import math
import time

import numpy as np
import PIL
from PIL import Image

from ._bicubic import resize_uint8_rgb


FACTOR = 32
MIN_PIXELS = 3136
MAX_PIXELS = 3145728
DEFAULT_IMAGE_PIXEL_LIMIT = 4_000_000
MAX_IMAGE_PIXEL_LIMIT = 6_291_456  # Same configurable upper bound as runtime/native/config.py.
MAX_BYTES = 128 * 1024 * 1024  # Same request bound as chandra_service/validation.py.
MEDIA = {"data:image/png;base64": "PNG", "data:image/jpeg;base64": "JPEG", "data:image/webp;base64": "WEBP"}
PROFILE = "qwen2-vl-transformers-5.18.0-aten-5c488690-uint8-aa-bicubic"


class PreparationError(ValueError):
    """Unsupported or oversized input, before image preparation or model submission."""


@dataclass(frozen=True)
class PreparedImage:
    png: bytes
    data_url: str
    metadata: dict


def _require(condition, message):
    if not condition:
        raise PreparationError(message)


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def smart_resize(width, height):
    """Pinned HF geometry, returning (width, height); this is not input admission.

    Transformers 5.18.0 Qwen2VLImageProcessor, Apache-2.0; see licenses/TRANSFORMERS.txt.
    """
    _require(type(width) is int and type(height) is int and width > 0 and height > 0, "Image dimensions must be positive integers")
    _require(max(width, height) / min(width, height) <= 200, "Image aspect ratio exceeds the processor limit of 200")
    h, w = round(height / FACTOR) * FACTOR, round(width / FACTOR) * FACTOR
    if h * w > MAX_PIXELS:
        beta = math.sqrt(height * width / MAX_PIXELS)
        h, w = max(FACTOR, math.floor(height / beta / FACTOR) * FACTOR), max(FACTOR, math.floor(width / beta / FACTOR) * FACTOR)
    elif h * w < MIN_PIXELS:
        beta = math.sqrt(MIN_PIXELS / (height * width))
        h, w = math.ceil(height * beta / FACTOR) * FACTOR, math.ceil(width * beta / FACTOR) * FACTOR
    return w, h


def _check_url_size(data_url):
    _require(isinstance(data_url, str), "Image data URL must be a string")
    _require(len(data_url) <= MAX_BYTES, "Image data URL exceeds the service's 128 MiB request bound")


def prepare_image(data_url, *, max_image_pixels=DEFAULT_IMAGE_PIXEL_LIMIT):
    """Decode one admitted still image, apply pinned resize, and return lossless PNG.

    max_image_pixels limits ORIGINAL decoded pixels; it never changes the processor profile.
    EXIF orientation, transparency and color profiles use the existing service's convert('RGB')
    semantics: no EXIF transpose, color management, crop or alpha compositing is introduced.
    """
    started = time.perf_counter()
    if np.__version__ != "2.5.3" or PIL.__version__ != "12.3.0":
        raise RuntimeError("This preparation profile requires NumPy 2.5.3 and Pillow 12.3.0")
    _require(type(max_image_pixels) is int and 1 <= max_image_pixels <= MAX_IMAGE_PIXEL_LIMIT, "Original pixel admission must be an integer in 1..6291456")
    _check_url_size(data_url)
    header, separator, encoded = data_url.partition(",")
    _require(separator and header in MEDIA, "Only PNG/JPEG/WebP base64 data URLs are supported")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except ValueError:
        raise PreparationError("Invalid image base64") from None
    _require(raw, "Empty image payload")
    try:
        with Image.open(io.BytesIO(raw), formats=list(MEDIA.values())) as original:
            _require(original.format == MEDIA[header], "Image bytes do not match the declared media type")
            _require(getattr(original, "n_frames", 1) == 1, "Only one still image is supported")
            original_size = original.size
            count = original.width * original.height
            _require(0 < count <= max_image_pixels, "Original image exceeds the configured decoded pixel limit")
            # Respect Pillow's warning threshold without changing process-global warning filters.
            _require(Image.MAX_IMAGE_PIXELS is None or count <= Image.MAX_IMAGE_PIXELS, "Image exceeds Pillow's decompression limit")
            size = smart_resize(*original_size)
            original.load()
            rgb = original.convert("RGB")
    except PreparationError:
        raise
    except (OSError, ValueError, SyntaxError, Image.DecompressionBombError, Image.DecompressionBombWarning):
        raise PreparationError("Invalid or oversized image payload") from None
    original_rgb = np.asarray(rgb, dtype=np.uint8)
    original_rgb_sha = _sha(original_rgb.tobytes())
    resize_started = time.perf_counter()
    prepared = resize_uint8_rgb(np, original_rgb, (size[1], size[0]))
    resize_ms = (time.perf_counter() - resize_started) * 1000
    png = io.BytesIO()
    Image.fromarray(prepared).save(png, format="PNG")
    png_bytes = png.getvalue()
    result_url = "data:image/png;base64," + base64.b64encode(png_bytes).decode("ascii")
    metadata = {
        "profile": PROFILE, "versions": {"numpy": np.__version__, "pillow": PIL.__version__},
        "original_format": MEDIA[header], "original_dimensions": list(original_size), "prepared_dimensions": list(size),
        "original_encoded_bytes": len(raw), "original_encoded_sha256": _sha(raw), "original_rgb_sha256": original_rgb_sha,
        "prepared_png_bytes": len(png_bytes), "prepared_png_sha256": _sha(png_bytes), "prepared_rgb_sha256": _sha(prepared.tobytes()),
        "original_pixel_admission_limit": max_image_pixels, "smart_resize": {"factor": FACTOR, "min_pixels": MIN_PIXELS, "max_pixels": MAX_PIXELS},
        "resized": size != original_size, "resize_ms": resize_ms, "total_ms": (time.perf_counter() - started) * 1000,
    }
    return PreparedImage(png_bytes, result_url, metadata)


def prepare_request(payload, *, max_image_pixels=DEFAULT_IMAGE_PIXEL_LIMIT):
    """Return (new request, PreparedImage), preserving every non-image field exactly.

    Missing optional fields remain missing. The caller owns full HTTP request validation, body
    admission and tokenizer/context checks. This function never clips an output allowance and
    does not import the service application merely to prepare an image.
    """
    _require(isinstance(payload, dict) and payload.get("model") == "chandra", "Use a request object with model alias chandra")
    tokens = payload.get("max_tokens", 12384)
    _require(type(tokens) is int and 1 <= tokens <= 12384, "max_tokens must be an integer in 1..12384")
    messages = payload.get("messages")
    _require(isinstance(messages, list) and len(messages) == 1, "Supply exactly one user message")
    message = messages[0]
    _require(isinstance(message, dict) and message.get("role") == "user", "Supply one user message")
    content = message.get("content")
    _require(isinstance(content, list) and len(content) == 2, "Supply one text and one image_url part")
    image_parts = [part for part in content if isinstance(part, dict) and part.get("type") == "image_url"]
    text_parts = [part for part in content if isinstance(part, dict) and part.get("type") == "text"]
    _require(len(image_parts) == 1 and len(text_parts) == 1, "Supply one text and one image_url part")
    text = text_parts[0].get("text")
    _require(isinstance(text, str) and 0 < len(text) <= 16384, "Text must be nonempty and at most 16384 characters")
    image_url = image_parts[0].get("image_url")
    _require(isinstance(image_url, dict), "image_url must contain a URL")
    url = image_url.get("url")
    _check_url_size(url)
    image = prepare_image(url, max_image_pixels=max_image_pixels)
    transformed = copy.deepcopy(payload)
    transformed_parts = [part for part in transformed["messages"][0]["content"] if part["type"] == "image_url"]
    transformed_parts[0]["image_url"]["url"] = image.data_url
    return transformed, image
