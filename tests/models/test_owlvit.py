"""OwLViT model tests and benchmarks."""
import pytest

from nos.models import OwlViT
from nos.test.utils import NOS_TEST_IMAGE, skip_if_no_torch_cuda


@pytest.fixture(scope="module")
def model():
    MODEL_NAME = "google/owlv2-base-patch16-ensemble"
    yield OwlViT(model_name=MODEL_NAME)


@skip_if_no_torch_cuda
def test_owlvit_caption(model):
    from PIL import Image

    im = Image.open(NOS_TEST_IMAGE)
    predictions = model(im, query=["bench", "car"])
    assert predictions is not None
    assert isinstance(predictions, dict)
    assert "scores" in predictions
    assert "labels" in predictions
    assert "boxes" in predictions
