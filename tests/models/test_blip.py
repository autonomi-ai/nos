"""BLIP model tests and benchmarks."""
import pytest

from nos.models import BLIP
from nos.test.utils import NOS_TEST_IMAGE, skip_if_no_torch_cuda


@pytest.fixture(scope="module")
def model():
    MODEL_NAME = "Salesforce/blip2-opt-2.7b"
    yield BLIP(model_name=MODEL_NAME)


@skip_if_no_torch_cuda
def test_blip_caption(model):
    from PIL import Image

    im = Image.open(NOS_TEST_IMAGE)
    caption = model(im)
    assert caption is not None

    vqa_answer = model(im, prompt="What's in this picture?")
    assert vqa_answer is not None
