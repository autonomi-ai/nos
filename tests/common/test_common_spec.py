from collections import namedtuple
from contextlib import contextmanager
from itertools import product
from typing import Dict, List, Tuple

import numpy as np
import pytest
from PIL import Image
from pydantic import ValidationError

from nos import hub
from nos.common.spec import FunctionSignature, ModelSpec
from nos.common.types import Batch, EmbeddingSpec, ImageSpec, ImageT, TensorSpec, TensorT
from nos.logging import logger


@contextmanager
def suppress_logger(name: str = __name__):
    logger.disable(name)
    yield
    logger.enable(name)


EMBEDDING_SHAPES = [
    (512,),
    (512,),
]
IMAGE_SHAPES = [
    (224, 224, 3),
    (224, 224, 3),
    (None, None, 3),
    (None, None, 3),
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
        spec = EmbeddingSpec(shape=shape, dtype=dtype)
        assert spec is not None


SigIO = namedtuple("SignatureInputOuput", ["input_annotations", "output_annotations"])


@pytest.fixture
def img2vec_signature():
    yield SigIO(
        input_annotations={"images": Batch[ImageT[Image.Image, ImageSpec(shape=(None, None, 3), dtype="uint8")]]},
        output_annotations={"embedding": Batch[TensorT[np.ndarray, EmbeddingSpec(shape=(512,), dtype="float32")]]},
    )


def test_common_model_spec(img2vec_signature):
    """Test model spec."""
    from typing import Union

    import numpy as np
    from PIL import Image

    from nos.protoc import import_module

    nos_service_pb2 = import_module("nos_service_pb2")

    class TestImg2VecModel:
        def __init__(self, model_name: str = "openai/clip"):
            self.model = hub.load(model_name)

        def __call__(self, img: Union[Image.Image, np.ndarray]) -> np.ndarray:
            embedding = self.model(img)
            return embedding

    # Model spec without any annotations
    spec = ModelSpec(
        "openai/clip",
        signature=FunctionSignature(
            func_or_cls=TestImg2VecModel,
            method="__call__",
        ),
    )
    assert spec is not None

    # Model spec with input / output annotations
    spec = ModelSpec(
        "openai/clip",
        signature=FunctionSignature(
            func_or_cls=TestImg2VecModel,
            method="__call__",
            input_annotations=img2vec_signature.input_annotations,
            output_annotations=img2vec_signature.output_annotations,
        ),
    )
    assert spec is not None
    assert spec.default_signature.func_or_cls is not None
    assert spec.default_signature.parameters is not None
    assert spec.default_signature.return_annotation is not None

    # Test serialization
    minfo = spec._to_proto()
    assert minfo is not None
    assert isinstance(minfo, nos_service_pb2.GenericResponse)

    spec_ = ModelSpec._from_proto(minfo)
    assert spec_ is not None
    assert isinstance(spec_, ModelSpec)
    assert spec_.default_signature.parameters is not None
    assert spec_.default_signature.return_annotation is not None
    assert (
        spec_.default_signature.func_or_cls is None
    ), "func_or_cls must be None since we do not allow serialization of custom models that have server-side dependencies"

    # Create a model spec with a wrong method name
    with pytest.raises((ValueError, ValidationError)):
        spec = ModelSpec(
            "openai/clip",
            signature=FunctionSignature(
                TestImg2VecModel,
                method="predict",
                input_annotations=img2vec_signature.input_annotations,
                output_annotations=img2vec_signature.output_annotations,
            ),
        )
        assert spec is not None

    # Test if ModelSpec initialization fails if the model id contains underscores or special characters
    for name in ["openai&clip", "openai\\clip", "openai:clip"]:
        with pytest.raises(ValueError):
            ModelSpec(
                name,
                signature=FunctionSignature(
                    TestImg2VecModel,
                    method="__call__",
                    input_annotations=img2vec_signature.input_annotations,
                    output_annotations=img2vec_signature.output_annotations,
                    init_args=("openai/clip",),
                    init_kwargs={},
                ),
            )


def test_common_model_spec_variations():
    # Create signatures for all tasks (without func_or_cls, init_args, init_kwargs, method)
    ImageSpec(shape=(None, None, 3), dtype="uint8")

    # Create custom class to test function signatures with different input/outputs
    class Custom:
        def embed_images(self, images: Image.Image) -> np.ndarray:
            return np.random.rand(512)

        def embed_images_dict(self, images: List[Image.Image]) -> Dict[str, np.ndarray]:
            return {"embedding": np.random.rand(512)}

        def embed_texts(self, texts: str) -> np.ndarray:
            return np.random.rand(512)

        def embed_texts_dict(self, texts: List[str]) -> Dict[str, np.ndarray]:
            return {"embedding": np.random.rand(512)}

        def img2bbox_tuple(self, images: Image.Image) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
            return np.random.rand(512), np.random.rand(512), np.random.rand(512)

        def img2bbox_dict(self, images: List[Image.Image]) -> Dict[str, np.ndarray]:
            return {
                "scores": np.random.rand(512),
                "labels": np.random.rand(512),
                "bboxes": np.random.rand(512),
            }

        def txt2img(self, texts: str) -> Image.Image:
            return Image.fromarray(np.random.rand(224, 224, 3).astype(np.uint8))

    # Image embedding (img2vec)
    img2vec_signature = FunctionSignature(
        Custom,
        method="embed_images",
        input_annotations={"images": Batch[ImageT[Image.Image, ImageSpec(shape=(None, None, 3), dtype="uint8")]]},
        output_annotations={"embedding": Batch[TensorT[np.ndarray, EmbeddingSpec(shape=(512,), dtype="float32")]]},
    )
    assert img2vec_signature is not None

    # Text embedding (txt2vec)
    txt2vec_signature = FunctionSignature(
        Custom,
        method="embed_texts",
        input_annotations={"texts": str},
        output_annotations={"embedding": Batch[TensorT[np.ndarray, EmbeddingSpec(shape=(512,), dtype="float32")]]},
    )
    assert txt2vec_signature is not None

    # Object detection (img2bbox)
    img2bbox_signature = FunctionSignature(
        Custom,
        method="img2bbox_dict",
        input_annotations={"images": Batch[ImageT[Image.Image, ImageSpec(shape=(None, None, 3), dtype="uint8")]]},
        output_annotations={
            "scores": Batch[TensorT[np.ndarray, TensorSpec(shape=(None), dtype="float32")]],
            "labels": Batch[TensorT[np.ndarray, TensorSpec(shape=(None), dtype="float32")]],
            "bboxes": Batch[TensorT[np.ndarray, TensorSpec(shape=(None, 4), dtype="float32")]],
        },
    )
    assert img2bbox_signature is not None

    # Image generation (txt2img)
    txt2img_signature = FunctionSignature(
        Custom,
        method="txt2img",
        input_annotations={"texts": Batch[str]},
        output_annotations={"images": Batch[ImageT[Image.Image, ImageSpec(shape=(None, None, 3), dtype="uint8")]]},
    )
    assert txt2img_signature is not None


def check_object_type(v):
    """Check if the object is of the correct type."""
    if isinstance(v, list):
        for item in v:
            check_object_type(item)
        return

    assert isinstance(v.is_batched(), bool)
    if v.batch_size():
        assert isinstance(v.batch_size(), int)
    assert v.base_type() is not None
    if v.base_spec():
        spec = v.base_spec()
        assert hasattr(spec, "shape")
        assert hasattr(spec, "dtype")

    if v.parameter is not None:
        assert v.parameter_name() is not None
        assert v.parameter_annotation() is not None


def test_common_spec_signature():
    """Test function signature."""

    for model_id in hub.list():
        spec: ModelSpec = hub.load_spec(model_id)
        logger.debug(f"{spec.name}, {spec.task}")
        assert spec is not None
        assert spec.name
        assert spec.task
        assert spec.default_signature.input_annotations is not None
        logger.debug(f"{spec.name}, {spec.task}")

        if isinstance(spec.default_signature.input_annotations, dict):
            for k, v in spec.default_signature.get_inputs_spec().items():
                logger.debug(f"input: {k}, {v}")
                check_object_type(v)
        if isinstance(spec.default_signature.output_annotations, dict):
            for k, v in spec.default_signature.get_outputs_spec().items():
                logger.debug(f"output: {k}, {v}")
                check_object_type(v)
        else:
            check_object_type(spec.default_signature.get_outputs_spec())


def test_common_spec_from_custom_model():
    """Test wrapping custom models for remote execution."""
    from typing import List, Union

    class CustomModel:
        """Custom inference model."""

        def __init__(self, model_name: str = "custom/model"):
            """Initialize the model."""
            from nos.logging import logger

            self.model_name = model_name
            logger.debug(f"Model {model_name} initialized.")

        def forward1(self):  # noqa: ANN001
            """Forward pass."""
            return True

        def forward2(self, images: Union[Image.Image, np.ndarray]) -> int:
            """Forward pass."""
            return images

        def __call__(
            self, images: Union[Image.Image, np.ndarray, List[Image.Image], List[np.ndarray]], n: int = 1
        ) -> List[int]:
            if (isinstance(images, np.ndarray) and images.ndim == 3) or isinstance(images, Image.Image):
                images = [images]
            return list(range(n * len(images)))

    # Get the model spec for remote execution
    CustomModel = ModelSpec.from_cls(CustomModel)
    assert CustomModel is not None
    assert isinstance(CustomModel, ModelSpec)

    # Get the default method
    method: str = CustomModel.default_method
    assert method is not None
    assert method == "__call__", "Default method must be __call__"

    # Get the function signature
    sig: FunctionSignature = CustomModel.default_signature
    assert sig is not None

    # Set the default method to be forward1
    CustomModel.set_default_method("forward1")
    # Check if the default method is set correctly
    # and that the cached_properties are re-computed
    assert CustomModel.default_method == "forward1"
    assert CustomModel.default_signature.method == "forward1"

    # Get model task
    # Note (spillai): This should raise a warning and we want to suppress it
    with suppress_logger("nos.common.spec"):
        task = CustomModel.task()
        assert task is None, "Custom models should not have a task unless explicitly added"

    # Get model spec metadata
    # Note (spillai): This should raise a warning and we want to suppress it
    with suppress_logger("nos.common.spec"):
        metadata = CustomModel.metadata()
        assert metadata.resources is None, "Custom models should not have resources set unless explicitly added"

    # Check if the wrapped model can be loaded (directly in the same process)
    model = CustomModel(model_name="custom/model")
    assert model is not None

    # Check if the model can be called
    images = [np.random.rand(224, 224, 3).astype(np.uint8)]
    result = model(images=images)
    assert result == [0]

    # Check if the model can be called with keyword arguments
    result = model(images=images, n=2)
    assert result == [0, 1]

    # Check if the model can be called with positional arguments
    result = model(images, 2)
    assert result == [0, 1]

    # Check if the model methods can be called
    result = model.forward1()
    assert result is True

    result = model.forward2(images)
    assert result is images


def test_model_spec_metadata():
    from nos.common import ModelResources, ModelSpecMetadata, ModelSpecMetadataCatalog

    catalog = ModelSpecMetadataCatalog.get()
    assert catalog is not None

    assert len(catalog._resources_catalog) > 0, "Resources catalog must not be empty"
    assert len(catalog._metadata_catalog) > 0, "Metadata catalog must not be empty"
    assert len(catalog._profile_catalog) > 0, "Profile catalog must not be empty"

    for k in catalog._metadata_catalog:
        metadata: ModelSpecMetadata = catalog._metadata_catalog[k]
        assert isinstance(metadata, ModelSpecMetadata)
        assert metadata.id is not None
        assert metadata.method is not None
        assert metadata.task is not None

        assert isinstance(metadata.resources, (type(None), ModelResources))
        assert metadata.profile is not None
        assert isinstance(metadata.profile, dict)
