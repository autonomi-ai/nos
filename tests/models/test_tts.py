from typing import Any

import pytest

from nos.test.utils import skip_if_no_torch_cuda


@pytest.fixture
def tts_model():
    from nos.models import TextToSpeech  # noqa: F401

    MODEL_NAME = "suno/bark-small"
    yield TextToSpeech(model_name=MODEL_NAME)


@skip_if_no_torch_cuda
def test_tts_generate(tts_model):
    transcription: Any = tts_model(
        prompts=["Ask not what your country can do for you, ask what you can do for your country"]
    )
    assert transcription is not None
