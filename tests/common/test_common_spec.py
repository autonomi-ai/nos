from itertools import product
from typing import List

import pytest
from pydantic import ValidationError

from nos import hub
from nos.common.spec import FunctionSignature, ModelSpec
from nos.common.tasks import TaskType
from nos.common.types import EmbeddingSpec, ImageSpec, TensorSpec


EMBEDDING_SHAPES = [
    (1, 512),
    (None, 512),
]
IMAGE_SHAPES = [
    (1, 224, 224, 3),
    (None, 224, 224, 3),
    (1, None, None, 3),
    (None, None, None, 3),
]
TENSOR_SHAPES = [
    *EMBEDDING_SHAPES,
    (1, 4, 4),
    (None, 4, 4),
    *IMAGE_SHAPES,
]
DTYPES = ["uint8", "int32", "int64", "float32", "float64"]


def test_common_tensor_spec_valid_shapes():
    """Test valid tensor shapes."""
    for shape, dtype in product(TENSOR_SHAPES, DTYPES):
        spec = TensorSpec(shape=shape, dtype=dtype)
    assert spec is not None


def test_common_tensor_spec_invalid_shapes():
    """Test invalid tensor shapes."""
    INVALID_SHAPES = [
        (1, 224, 224, 3, 3),
        (None, 224, 224, 3, 3),
        (1, 224, 224, 1, 2, 3),
        (None, 224, 224, 1, 2, 3),
        (1,),
        (None,),
        (None, None, None, None, None),
    ]
    for shape in INVALID_SHAPES:
        with pytest.raises(ValueError):
            spec = TensorSpec(shape=shape, dtype="uint8")
            assert spec is not None


def test_common_image_spec_valid_shapes():
    """Test valid image shapes."""
    for shape, dtype in product(IMAGE_SHAPES, DTYPES):
        spec = ImageSpec(shape=shape, dtype=dtype)
        assert spec is not None


def test_common_embedding_spec_valid_shapes():
    """Test valid embedding shapes."""
    for shape, dtype in product(EMBEDDING_SHAPES, DTYPES):
        spec = ImageSpec(shape=shape, dtype=dtype)
        assert spec is not None


@pytest.fixture
def img2vec_signature():
    yield FunctionSignature(
        inputs={"img": ImageSpec(shape=(None, None, None, 3), dtype="uint8")},
        outputs={"embedding": EmbeddingSpec(shape=(None, 512), dtype="float32")},
    )


def test_common_model_spec(img2vec_signature):
    """Test model spec."""
    from typing import Union

    import numpy as np
    from PIL import Image

    class TestImg2VecModel:
        def __init__(self, model_name: str = "openai/clip"):
            self.model = hub.load(model_name)

        def __call__(self, img: Union[Image.Image, np.ndarray]) -> np.ndarray:
            embedding = self.model(img)
            return embedding

    spec = ModelSpec(
        name="openai/clip",
        task=TaskType.IMAGE_EMBEDDING,
        signature=FunctionSignature(
            inputs=img2vec_signature.inputs,
            outputs=img2vec_signature.outputs,
            func_or_cls=TestImg2VecModel,
            init_args=("openai/clip",),
            init_kwargs={},
            method_name="__call__",
        ),
    )
    assert spec is not None

    # Create a model spec with a wrong method name
    with pytest.raises(ValidationError):
        spec = ModelSpec(
            name="openai/clip",
            task=TaskType.IMAGE_EMBEDDING,
            signature=FunctionSignature(
                inputs=img2vec_signature.inputs,
                outputs=img2vec_signature.outputs,
                func_or_cls=TestImg2VecModel,
                init_args=("openai/clip",),
                init_kwargs={},
                method_name="predict",
            ),
        )
        assert spec is not None


def test_common_model_spec_variations():
    # Create signatures for all tasks (without func_or_cls, init_args, init_kwargs, method_name)
    ImageSpec(shape=(None, None, None, 3), dtype="uint8")

    # Image embedding (img2vec)
    img2vec_signature = FunctionSignature(
        inputs={"images": ImageSpec(shape=(None, None, None, 3), dtype="uint8")},
        outputs={"embedding": EmbeddingSpec(shape=(None, 512), dtype="float32")},
    )
    assert img2vec_signature is not None

    # Text embedding (txt2vec)
    txt2vec_signature = FunctionSignature(
        inputs={"texts": List[str]},
        outputs={"embedding": EmbeddingSpec(shape=(None, 512), dtype="float32")},
    )
    assert txt2vec_signature is not None

    # Object detection (img2bbox)
    img2bbox_signature = FunctionSignature(
        inputs={"images": ImageSpec(shape=(None, None, None, 3), dtype="uint8")},
        outputs={
            "scores": TensorSpec(shape=(None, None), dtype="uint8"),
            "labels": TensorSpec(shape=(None, None), dtype="uint8"),
            "bboxes": TensorSpec(shape=(None, None), dtype="uint8"),
        },
    )
    assert img2bbox_signature is not None

    # Image generation (txt2img)
    txt2img_signature = FunctionSignature(
        inputs={"texts": List[str]},
        outputs={"images": ImageSpec(shape=(None, None, None, 3), dtype="uint8")},
    )
    assert txt2img_signature is not None
