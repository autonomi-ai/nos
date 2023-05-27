import typing
from typing import Generic, Optional, Tuple, TypeVar

import numpy as np
from pydantic import ValidationError, validator
from pydantic.dataclasses import dataclass
from typing_extensions import Annotated


T = TypeVar("T")

# TODO (spillai): Type aliases
# Image = Union[PIL.Image.Image, np.ndarray]
# Tensor = Union[np.ndarray, torch.Tensor]
# Embedding = Tensor


@dataclass(frozen=True)
class TensorSpec:
    shape: Optional[Tuple[Optional[int], ...]] = None
    """Base tensor specification with at most 4 dimensions.

    This class is used to capture the shape and dtype of a tensor.
    Tensor shapes are specified as a tuple of integers, where
    the first dimension is the batch size. Each of the dimensions
    are optional, and can be set to None to support dynamic dimension.
    For e.g., a model that supports variable batch size has
    shape=(None, 224, 224, 3).

    Examples:
        ImageSpec:
            - (H, W, C): (height, width, channels)
        EmbeddingSpec:
            - (D): (dims, )
    """
    dtype: str = None
    """Tensor dtype. (uint8, int32, int64, float32, float64)"""

    @validator("shape")
    def validate_shape(cls, shape: Optional[Tuple[Optional[int], ...]]):
        """Validate the shape."""
        if shape and (len(shape) < 1 or len(shape) > 4):
            raise ValidationError(f"Invalid tensor shape [shape={shape}].")
        else:
            return shape

    @validator("dtype")
    def validate_dtype(cls, dtype: str):
        """Validate the dtype."""
        if dtype and not hasattr(np, dtype):
            raise ValidationError(f"Invalid dtype [dtype={dtype}].")
        else:
            return dtype


@dataclass(frozen=True)
class ImageSpec(TensorSpec):
    """Image tensor specification with dimensions (H, W, C)."""

    @validator("shape")
    def validate_shape(cls, shape: Tuple[Optional[int], ...]):
        """Validate the shape."""
        if shape and len(shape) != 3:
            raise ValidationError(f"Invalid image shape [shape={shape}].")
        else:
            return shape


@dataclass(frozen=True)
class EmbeddingSpec(TensorSpec):
    """Embedding tensor specification with dimensions (D)."""

    @validator("shape")
    def validate_shape(cls, shape: Tuple[Optional[int]]):
        """Validate the shape."""
        if shape and len(shape) != 1:
            raise ValidationError(f"Invalid embedding shape [shape={shape}].")
        else:
            return shape


class Batch(Generic[T]):
    """Generic annotation/type-hint for batched data.

    Inherits from typing.Annotated[T, x] (PEP 593) where T is the type,
    and x is the metadata. The metadata is tpyically ignored,
    but can be used to allow additional type checks and annotations
    on the type.
    """

    __slots__ = ()

    @typing._tp_cache
    def __class_getitem__(cls, params):
        """Support Batch[T, batch_size].

        Annotated requires atleast 2 parameters [type, metadata].
        Here `batch_size` is optional (i.e. Batch[T],
        is equivalent to Batch[T, None]).
        """
        if not isinstance(params, tuple):
            params = (params, None)
        _, batch_size = params
        if batch_size is not None:
            if isinstance(batch_size, int):
                if batch_size < 1 or batch_size >= 65536:
                    raise ValueError(f"Invalid batch size [batch_size={batch_size}].")
            else:
                raise TypeError(f"Invalid batch size type [type(batch_size)={type(batch_size)}].")
        return Annotated[cls, params]


class TensorT(Generic[T]):
    """Generic annotation/type-hint for batched data.

    Inherits from typing.Annotated[T, x] (PEP 593) where T is the type,
    and x is the metadata. The metadata is tpyically ignored,
    but can be used to allow additional type checks and annotations
    on the type.
    """

    __slots__ = ()

    @typing._tp_cache
    def __class_getitem__(cls, params):
        """Support TensorT[type, tensor_spec].

        Annotated requires atleast 2 parameters [type, tensor_spec].
        Here `tensor_spec` is optional (i.e. TensorT[T],
        is equivalent to TensorT[T, None]).
        """
        if not isinstance(params, tuple):
            params = (params, TensorSpec())
        _, tensor_spec = params
        if tensor_spec is not None:
            if not isinstance(tensor_spec, TensorSpec):
                raise TypeError(f"Invalid tensor_spec metadata [tensor_spec={type(tensor_spec)}].")
        return Annotated[cls, params]


ImageT = TensorT
