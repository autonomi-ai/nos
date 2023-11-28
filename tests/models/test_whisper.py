from pathlib import Path
from typing import Any, Dict, List

import pytest

from nos.test.conftest import model_manager as manager  # noqa: F401, F811
from nos.test.utils import NOS_TEST_AUDIO, skip_if_no_torch_cuda


@pytest.fixture
def whisper_model():
    from nos.models import Whisper  # noqa: F401

    MODEL_NAME = "openai/whisper-small.en"
    yield Whisper(model_name=MODEL_NAME)


@skip_if_no_torch_cuda
def test_whisper_transcribe_audio_file(whisper_model):
    audio_path = Path(NOS_TEST_AUDIO)
    assert audio_path.exists()

    transcription: List[Dict[str, Any]] = whisper_model.transcribe(audio_path)
    assert transcription is not None
    for item in transcription:
        assert "timestamp" in item
        assert "text" in item
