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


@pytest.fixture
def whisperx_model(manager):  # noqa: F811
    from nos.common import ModelSpec
    from nos.managers import ModelHandle
    from nos.models import WhisperX  # noqa: F401

    MODEL_NAME = "m-bain/whisperx-large"
    cfg = WhisperX.configs[MODEL_NAME]

    # Get the model spec for remote execution
    spec = ModelSpec.from_cls(
        WhisperX,
        init_args=(),
        init_kwargs={"model_name": MODEL_NAME},
        runtime_env=cfg.runtime_env,
        method_name="transcribe_file",
    )
    assert spec is not None
    assert isinstance(spec, ModelSpec)

    # Check if the model can be loaded with the ModelManager
    # Note: This will be executed as a Ray actor within a custom runtime env.
    model_handle = manager.load(spec)
    assert model_handle is not None
    assert isinstance(model_handle, ModelHandle)

    yield model_handle


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


@pytest.mark.skip("TOFIX (spillai): Pending whisperx integration with nos.")
@pytest.mark.server
def test_whisperx_transcribe_audio_file(manager, whisperx_model):  # noqa: F811
    Path(NOS_TEST_AUDIO)

    transcription = whisperx_model(filename=str(NOS_TEST_AUDIO))
    assert transcription is not None
    for item in transcription:
        assert "timestamp" in item
        assert "text" in item
