from typing import TypeVar, Union, get_args

import numpy as np
import PIL
import pytest

from nos.common import EmbeddingSpec, ImageSpec, TensorSpec
from nos.common.types import Batch, ImageT, TensorT


try:
    import torch

    has_torch = True
except ImportError:
    has_torch = False


def test_common_types():
    import numpy as np
    from PIL import Image

    # Tensor types with TensorSpec metadata
    TensorT[np.ndarray]  # without metadata (just a dyanmic tensor)
    TensorT[np.ndarray, TensorSpec(shape=(480, 640, 3), dtype="float32")]
    TensorT[np.ndarray, TensorSpec(shape=(None, None, 3), dtype="float32")]
    if has_torch:
        TensorT[torch.Tensor]
        TensorT[torch.Tensor, TensorSpec(shape=(None, None, 3), dtype="float32")]
        TensorT[torch.Tensor, TensorSpec(shape=(480, 640, 3), dtype="float32")]

    # Embedding types (subclass of Tensor) with EmbeddingSpec metadata
    TensorT[np.ndarray, EmbeddingSpec(shape=(512,), dtype="float32")]
    TensorT[np.ndarray, EmbeddingSpec(shape=(None,), dtype="float16")]
    if has_torch:
        TensorT[torch.Tensor, EmbeddingSpec(shape=(512,), dtype="float32")]

    # Image types (subclass of Tensor) with ImageSpec metadata
    ImageT[np.ndarray]  #  without metadata (just a dyanmic image)
    ImageT[PIL.Image.Image]
    ImageT[Image.Image, ImageSpec(shape=(None, None, 3), dtype="float32")]  # with channel metadata, dtype=float32
    ImageT[np.ndarray, ImageSpec(shape=(480, 640, 3), dtype="uint8")]  # with full shape metadata, dtype=uint8
    if has_torch:
        ImageT[torch.Tensor]


def test_common_batch_types():
    T = TypeVar("T")

    Batch[str]
    Batch[np.ndarray]
    Batch[PIL.Image.Image]
    Batch[T]

    b1 = Batch[ImageT[np.ndarray]]  # equivalent to Batch[np.ndarray, None]: dynamic batch, dynamic shape
    b2 = Batch[ImageT[np.ndarray], None]  # (None, ...): dyanmic batch, dynamic shape
    b3 = Batch[ImageT[np.ndarray], 8]  # (   8, ...): fixed batch, dynamic shape
    b4 = Batch[
        ImageT[np.ndarray, ImageSpec(shape=(None, None, 3), dtype="uint8")], 8
    ]  # (8, None, None, 3): fixed batch, dynamic shape, fixed channel
    b5 = Batch[
        ImageT[np.ndarray, ImageSpec(shape=(480, 640, 3), dtype="uint8")], 8
    ]  # (8, 480, 640, 3): fixed batch, fixed shape, fixed channel

    b6 = Batch[ImageT[PIL.Image.Image]]
    b7 = Batch[ImageT[PIL.Image.Image], 8]

    Image = Union[np.ndarray, PIL.Image.Image]
    Batch[ImageT[Image]]
    Batch[ImageT[Image], 8]
    Batch[ImageT[Image, ImageSpec(shape=(None, None, 3), dtype="uint8")], 8]
    Batch[ImageT[Image, ImageSpec(shape=(480, 640, 3), dtype="uint8")], 8]

    Batch[TensorT[np.ndarray]]
    Batch[TensorT[np.ndarray], 8]
    Batch[TensorT[np.ndarray, EmbeddingSpec(shape=(None,), dtype="float32")], 8]
    Batch[TensorT[np.ndarray, EmbeddingSpec(shape=(512,), dtype="float32")], 8]

    if has_torch:
        Batch[ImageT[torch.Tensor]]

    for btype in [b1, b2, b3, b4, b5, b6, b7]:
        assert len(btype.__metadata__) == 1
        tensor_type, batch_size = btype.__metadata__[0]
        assert batch_size is None or batch_size > 0
        assert tensor_type.__metadata__[0][0] in get_args(Union[np.ndarray, PIL.Image.Image])
        assert isinstance(tensor_type.__metadata__[0][1], TensorSpec)

    # Batch validation
    INVALID_BATCH_SIZES = [0, 65537, -1, "8", 1.0]
    for b in INVALID_BATCH_SIZES:
        with pytest.raises(Exception):
            Batch[T, b]
    with pytest.raises(Exception):
        Batch[T, None, None]
