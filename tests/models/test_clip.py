import pytest
from PIL import Image

from nos.models import CLIP
from nos.test.benchmark import run_benchmark
from nos.test.utils import NOS_TEST_IMAGE, PyTestGroup, benchmark, requires_torch_cuda


pytestmark = pytest.mark.skipif(not requires_torch_cuda, reason="Requires CUDA")


MODEL_NAME = "openai/clip-vit-base-patch32"


@pytest.fixture(scope="module")
def model():
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


@requires_torch_cuda
def test_clip_encode_text(model):
    _test_clip_encode_text(model)


@requires_torch_cuda
def test_clip_encode_image(model):
    _test_clip_encode_image(model)


@benchmark
@requires_torch_cuda
@pytest.mark.benchmark(group=PyTestGroup.HUB)
def test_clip_model_variants():
    model = CLIP(model_name="openai/clip-vit-base-patch32")
    _test_clip_encode_text(model, D=512),
    _test_clip_encode_image(model, D=512)

    model = CLIP(model_name="openai/clip-vit-large-patch14")
    _test_clip_encode_text(model, D=768),
    _test_clip_encode_image(model, D=768)

    model = CLIP(model_name="laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
    _test_clip_encode_text(model, D=1024),
    _test_clip_encode_image(model, D=1024)

    model = CLIP(model_name="laion/CLIP-ViT-L-14-laion2B-s32B-b82K")
    _test_clip_encode_text(model, D=768),
    _test_clip_encode_image(model, D=768)


@benchmark
@requires_torch_cuda
@pytest.mark.benchmark(group=PyTestGroup.BENCHMARK_MODELS)
@pytest.mark.parametrize(
    "model_name",
    [
        "openai/clip-vit-base-patch32",
        "openai/clip-vit-large-patch14",
        "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
        "laion/CLIP-ViT-L-14-laion2B-s32B-b82K",
    ],
)
def test_clip_visual_benchmark(model_name):
    """
    Benchmark results (NVIDIA GeForce RTX 2080 Ti):
    - [openai/clip-vit-base-patch32]: 9.98 ms / step
    - [openai/clip-vit-large-patch14]: 23.84 ms / step
    - [laion/CLIP-ViT-H-14-laion2B-s32B-b79K]: 51.53 ms / step
    - [laion/CLIP-ViT-L-14-laion2B-s32B-b82K]: 24.50 ms / step
    """

    img = Image.open(NOS_TEST_IMAGE)

    model = CLIP(model_name=model_name)
    time_ms = run_benchmark(
        lambda: model.encode_image(img),
        num_iters=1000,
    )
    print(f"BENCHMARK [{model_name}]: {time_ms:.2f} ms / step")
