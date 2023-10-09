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
    audio_file = Path(NOS_TEST_AUDIO)
    assert audio_file.exists()

    transcription: List[Dict[str, Any]] = whisper_model.transcribe_file(filename=str(audio_file))
    assert transcription is not None
    for item in transcription:
        assert "timestamp" in item
        assert "text" in item


@skip_if_no_torch_cuda
def test_whisper_transcribe_audio_blob(whisper_model):
    audio_file = Path(NOS_TEST_AUDIO)
    assert audio_file.exists()
    with open(audio_file, "rb") as f:
        audio_data = f.read()

    import base64

    audio_data_base64 = base64.b64encode(audio_data).decode("utf-8")

    transcription: List[Dict[str, Any]] = whisper_model.transcribe_file_blob(audio_data_base64)
    assert transcription is not None
    for item in transcription:
        assert "timestamp" in item
        assert "text" in item
