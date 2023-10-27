import base64
import json
from io import BytesIO
from pathlib import Path
from typing import Any, Dict

import numpy as np
from fastapi import UploadFile
from fastapi.responses import FileResponse
from PIL import Image


def encode_item(v: Any) -> Any:
    """Encode an item to a JSON-serializable object."""
    if isinstance(v, dict):
        return {k: encode_item(_v) for k, _v in v.items()}
    elif isinstance(v, (list, tuple, set, frozenset)):
        return [encode_item(x) for x in v]
    elif isinstance(v, Image.Image):
        return image_to_base64_str(v)
    elif isinstance(v, np.ndarray):
        if v.ndim <= 2:
            return v.tolist()
        else:
            raise ValueError(f"Unsupported ndarray dimension: {v.ndim}")
    elif isinstance(v, Path):
        return FileResponse(v)
    else:
        return v


def decode_item(v: Any) -> Any:
    """Decode an item from a JSON-serializable object."""
    if isinstance(v, dict):
        return {k: decode_item(_v) for k, _v in v.items()}
    elif isinstance(v, (list, tuple, set, frozenset)):
        return [decode_item(x) for x in v]
    elif isinstance(v, str) and v.startswith("data:image/"):
        return base64_str_to_image(v)
    else:
        return v


encode_dict = encode_item
decode_dict = decode_item


def _decode_file_object(file_object: UploadFile) -> Dict[str, Any]:
    """Decode a file object to a dictionary."""
    if file_object.content_type == "application/json":
        return decode_dict(json.load(file_object.file))
    elif file_object.content_type == "application/x-www-form-urlencoded":
        return decode_dict(json.loads(file_object.file.read()))
    elif file_object.content_type == "image/jpeg":
        return {"images": Image.open(BytesIO(file_object.file.read())).convert("RGB")}
    else:
        raise ValueError(f"Unknown content type: {file_object.content_type}")


def image_to_base64_str(image: Image.Image, format: str = "PNG") -> str:
    """Convert an image to a base64 string."""
    if format not in ("PNG", "JPEG"):
        raise ValueError(f"Unsupported image format: {format}")
    buffered = BytesIO()
    image_format = image.format or format
    image.save(buffered, format=image_format)
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/{image_format.lower()};base64,{img_str}"


def base64_str_to_image(base64_str: str) -> Image.Image:
    """Convert a base64 string to an image."""
    base64_str = base64_str.strip()
    if not base64_str.startswith("data:image/"):
        raise ValueError(f"Invalid base64 string: {base64_str}")
    _, base64_str = base64_str.split(";base64,")
    return Image.open(BytesIO(base64.b64decode(base64_str))).convert("RGB")


class CustomJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder."""

    def default(self, obj):
        return encode_dict(obj)
