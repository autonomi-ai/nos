"""CLIP model tests and benchmarks.

Benchmark results (NVIDIA GeForce RTX 2080 Ti):
- [openai/clip-vit-base-patch32]: 9.98 ms / step
- [openai/clip-vit-large-patch14]: 23.84 ms / step
- [laion/CLIP-ViT-H-14-laion2B-s32B-b79K]: 51.53 ms / step
- [laion/CLIP-ViT-L-14-laion2B-s32B-b82K]: 24.50 ms / step
"""
import pytest
from PIL import Image

from nos.common import tqdm
from nos.logging import logger
from nos.models import CLIP
from nos.test.utils import NOS_TEST_IMAGE, PyTestGroup, skip_if_no_torch_cuda


@pytest.fixture(scope="module")
def model():
    MODEL_NAME = "openai/clip-vit-base-patch32"
    yield CLIP(model_name=MODEL_NAME)


def _test_clip_encode_text(_model, D: int = 512):
    embed_text = _model.encode_text("text")
    assert embed_text.shape == (1, D)
    embed_text = _model.encode_text(["text"])
    assert embed_text.shape == (1, D)
    embed_texts = _model.encode_text(["text", "more"])
    assert embed_texts.shape == (2, D)


def _test_clip_encode_image(_model, D: int = 512):
    from PIL import Image

    im = Image.open(NOS_TEST_IMAGE)
    embed_im = _model.encode_image(im)
    assert embed_im.shape == (1, D)
    embed_im = _model.encode_image([im])
    assert embed_im.shape == (1, D)
    embed_im = _model.encode_image([im, im.rotate(180)])
    assert embed_im.shape == (2, D)


@pytest.mark.benchmark(group=PyTestGroup.HUB)
def test_clip_encode_text(model):
    _test_clip_encode_text(model)


@pytest.mark.benchmark(group=PyTestGroup.HUB)
def test_clip_encode_image(model):
    _test_clip_encode_image(model)


@pytest.mark.benchmark(group=PyTestGroup.HUB)
def test_clip_model_variants():
    """Test all CLIP model variants."""
    for model_name in CLIP.configs.keys():
        model = CLIP(model_name=model_name)
        _test_clip_encode_text(model, D=model.cfg.D)
        _test_clip_encode_image(model, D=model.cfg.D)


@skip_if_no_torch_cuda
@pytest.mark.benchmark(group=PyTestGroup.MODEL_BENCHMARK)
@pytest.mark.parametrize(
    "model_name",
    [
        "openai/clip-vit-base-patch32",
        # "openai/clip-vit-large-patch14",
        # "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
        # "laion/CLIP-ViT-L-14-laion2B-s32B-b82K",
    ],
)
def test_clip_visual_benchmark(model_name):
    """Benchmark CLIP visual encoder."""
    import numpy as np

    img = Image.open(NOS_TEST_IMAGE)
    img = img.resize((224, 224))

    model = CLIP(model_name=model_name)
    logger.debug(f"Benchmarking {model_name}")
    logger.debug("Warming up (5s) ...")
    for _ in tqdm(duration=5.0, disable=True):
        model.encode_image([img])
    logger.debug("Running benchmark (20s) ...")
    B = 128
    images = [np.asarray(img) for _ in range(B)]
    for _ in tqdm(duration=20.0, unit_scale=B, desc=f"torch-eager | {model_name}"):
        model.encode_image(images)
