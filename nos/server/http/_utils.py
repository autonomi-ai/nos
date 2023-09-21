import base64
import json
from io import BytesIO
from typing import Any, Dict

import numpy as np
from fastapi import UploadFile
from PIL import Image


def encode_item(v: Any) -> Any:
    """Encode an item to a JSON-serializable object."""
    if isinstance(v, Image.Image):
        return image_to_base64_str(v)
    elif isinstance(v, np.ndarray):
        return v.tolist()
    elif isinstance(v, (list, tuple)):
        return [encode_item(x) for x in v]
    elif isinstance(v, dict):
        return {k: encode_item(_v) for k, _v in v.items()}
    else:
        return v


def decode_item(v: Any) -> Any:
    """Decode an item from a JSON-serializable object."""
    if isinstance(v, str) and v.startswith("data:image/"):
        return Image.open(BytesIO(base64.b64decode(v.split(",")[1]))).convert("RGB")
    elif isinstance(v, (list, tuple)):
        return [decode_item(x) for x in v]
    elif isinstance(v, dict):
        return {k: decode_item(_v) for k, _v in v.items()}
    else:
        return v


def encode_dict(d: Any) -> Dict[str, Any]:
    """Encode a dictionary to a JSON-serializable object."""
    return {k: encode_item(v) for k, v in d.items()}


def decode_dict(d: Any) -> Dict[str, Any]:
    """Decode a dictionary from a JSON-serializable object."""
    return {k: decode_item(v) for k, v in d.items()}


def decode_file_object(file_object: UploadFile) -> Dict[str, Any]:
    """Decode a file object to a dictionary."""
    if file_object.content_type == "application/json":
        return decode_dict(json.load(file_object.file))
    elif file_object.content_type == "application/x-www-form-urlencoded":
        return decode_dict(json.loads(file_object.file.read()))
    elif file_object.content_type == "image/jpeg":
        return {"images": Image.open(BytesIO(file_object.file.read())).convert("RGB")}
    else:
        raise ValueError(f"Unknown content type: {file_object.content_type}")


def image_to_base64_str(image: Image.Image) -> str:
    """Convert an image to a base64 string."""
    buffered = BytesIO()
    image_format = image.format or "PNG"
    image.save(buffered, format=image_format)
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/{image_format.lower()};base64,{img_str}"


class CustomJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder."""

    def default(self, obj):
        return encode_dict(obj)
