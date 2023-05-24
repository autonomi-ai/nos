from typing import Optional, Tuple

from pydantic import ValidationError, validator
from pydantic.dataclasses import dataclass


@dataclass(frozen=True)
class TensorSpec:
    shape: Tuple[Optional[int], ...]
    """Base tensor specification with at most 4 dimensions.

    This class is used to capture the shape and dtype of a tensor.
    Tensor shapes are specified as a tuple of integers, where
    the first dimension is the batch size. Each of the dimensions
    are optional, and can be set to None to support dynamic dimension.
    For e.g., a model that supports variable batch size has
    shape=(None, 224, 224, 3).

    Examples:
        ImageSpec:
            - (B, H, W, C): (batch_size, height, width, channels)
        EmbeddingSpec:
            - (B, D): (batch_size, dims)
    """
    dtype: str
    """Tensor dtype. (uint8, int32, int64, float32, float64)"""

    @validator("shape")
    def validate_shape(cls, shape: Tuple[Optional[int], ...]):
        """Validate the shape."""
        if len(shape) <= 1 or len(shape) > 4:
            raise ValidationError(f"Invalid tensor shape [shape={shape}].")
        else:
            return shape

    @validator("dtype")
    def validate_dtype(cls, dtype: str):
        """Validate the dtype."""
        if dtype not in ("uint8", "int32", "int64", "float32", "float64"):
            raise ValidationError(f"Invalid dtype [dtype={dtype}].")
        else:
            return dtype


class ImageSpec(TensorSpec):
    """Image tensor specification with dimensions (B, H, W, C)."""

    @validator("shape")
    def validate_shape(cls, shape: Tuple[Optional[int], ...]):
        """Validate the shape."""
        if len(shape) != 4:
            raise ValidationError(f"Invalid image shape [shape={shape}].")
        else:
            return shape


class EmbeddingSpec(TensorSpec):
    """Embedding tensor specification with dimensions (B, D)."""

    @validator("shape")
    def validate_shape(cls, shape: Tuple[Optional[int], ...]):
        """Validate the shape."""
        if len(shape) != 2:
            raise ValidationError(f"Invalid embedding shape [shape={shape}].")
        else:
            return shape
