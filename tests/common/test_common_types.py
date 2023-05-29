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
    Batch[str, None]
    Batch[str, 8]

    Batch[np.ndarray]
    Batch[np.ndarray, None]
    Batch[np.ndarray, 8]

    Batch[PIL.Image.Image]
    Batch[T]

    tensor_b1 = Batch[TensorT[np.ndarray]]
    tensor_b2 = Batch[TensorT[np.ndarray], 8]
    tensor_b3 = Batch[TensorT[np.ndarray, EmbeddingSpec(shape=(None,), dtype="float32")], 8]
    tensor_b4 = Batch[TensorT[np.ndarray, EmbeddingSpec(shape=(512,), dtype="float32")], 8]

    if has_torch:
        Batch[ImageT[torch.Tensor]]
        Batch[ImageT[torch.Tensor], 8]

    img_b1 = Batch[ImageT[np.ndarray]]  # equivalent to Batch[np.ndarray, None]: dynamic batch, dynamic shape
    img_b2 = Batch[ImageT[np.ndarray], None]  # (None, ...): dyanmic batch, dynamic shape
    img_b3 = Batch[ImageT[np.ndarray], 8]  # (   8, ...): fixed batch, dynamic shape
    img_b4 = Batch[
        ImageT[np.ndarray, ImageSpec(shape=(None, None, 3), dtype="uint8")], 8
    ]  # (8, None, None, 3): fixed batch, dynamic shape, fixed channel
    img_b5 = Batch[
        ImageT[np.ndarray, ImageSpec(shape=(480, 640, 3), dtype="uint8")], 8
    ]  # (8, 480, 640, 3): fixed batch, fixed shape, fixed channel

    img_b6 = Batch[ImageT[PIL.Image.Image]]
    img_b7 = Batch[ImageT[PIL.Image.Image], 8]

    Image = Union[np.ndarray, PIL.Image.Image]
    Batch[ImageT[Image]]
    Batch[ImageT[Image], 8]
    Batch[ImageT[Image, ImageSpec(shape=(None, None, 3), dtype="uint8")], 8]
    Batch[ImageT[Image, ImageSpec(shape=(480, 640, 3), dtype="uint8")], 8]

    # Various tensor types
    for btype in [tensor_b1, tensor_b2, tensor_b3, tensor_b4]:
        assert len(btype.__args__) == 1
        assert len(btype.__metadata__) == 2

        (batch_cls,) = btype.__args__
        annotated_tensor, batch_size = btype.__metadata__
        assert batch_cls == Batch
        assert batch_size is None or batch_size > 0

        (tensor_cls,) = annotated_tensor.__args__
        tensor_type, tensor_spec = annotated_tensor.__metadata__
        assert tensor_cls == TensorT
        assert tensor_type == np.ndarray
        assert isinstance(tensor_spec, TensorSpec)

    # Various image types
    for btype in [img_b1, img_b2, img_b3, img_b4, img_b5, img_b6, img_b7]:
        assert len(btype.__args__) == 1
        assert len(btype.__metadata__) == 2

        (batch_cls,) = btype.__args__
        annotated_image, batch_size = btype.__metadata__
        assert batch_cls == Batch
        assert batch_size is None or batch_size > 0

        (image_cls,) = annotated_image.__args__
        image_type, image_spec = annotated_image.__metadata__
        assert image_cls == ImageT
        assert image_type in get_args(Union[np.ndarray, PIL.Image.Image])
        assert isinstance(image_spec, ImageSpec)

    # Batch validation
    INVALID_BATCH_SIZES = [0, 65537, -1, "8", 1.0]
    for b in INVALID_BATCH_SIZES:
        with pytest.raises(Exception):
            Batch[T, b]
    with pytest.raises(Exception):
        Batch[T, None, None]
