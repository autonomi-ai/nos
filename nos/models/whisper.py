import base64
import io
import os
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List

import torch

from nos import hub
from nos.common import TaskType
from nos.common.types import Batch
from nos.hub import HuggingFaceHubConfig
from nos.logging import logger


@dataclass(frozen=True)
class WhisperConfig(HuggingFaceHubConfig):

    chunk_length_s: int = 30
    """Chunk length in seconds."""


class Whisper:
    """Whisper model for audio transcription.
    https://huggingface.co/docs/transformers/model_doc/whisper
    """

    configs = {
        "openai/whisper-base.en": WhisperConfig(
            model_name="openai/whisper-base.en",
        ),
        "openai/whisper-tiny.en": WhisperConfig(
            model_name="openai/whisper-tiny.en",
        ),
        "openai/whisper-small.en": WhisperConfig(
            model_name="openai/whisper-small.en",
        ),
        "openai/whisper-medium.en": WhisperConfig(
            model_name="openai/whisper-medium.en",
        ),
        "openai/whisper-large-v2": WhisperConfig(
            model_name="openai/whisper-large-v2",
        ),
    }

    def __init__(self, model_name: str = "openai/whisper-tiny.en"):

        logger.info("Init whisper large")
        from transformers import pipeline

        try:
            self.cfg = Whisper.configs[model_name]
        except KeyError:
            raise ValueError(f"Invalid model_name: {model_name}, available models: {Whisper.configs.keys()}")

        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            torch.backends.cuda.matmul.allow_tf32 = True
        else:
            self.device = torch.device("cpu")

        model_name = self.cfg.model_name
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model_name,
            chunk_length_s=self.cfg.chunk_length_s,
            device=self.device,
        )

    def transcribe_file(self, filename: str) -> List[Dict[str, Any]]:
        """Transcribe the audio file."""
        with torch.inference_mode():
            # Run the prediction
            # prediction = [{'text': ' ...', 'timestamp': (0.0, 5.44)}]
            return self.pipe(filename, return_timestamps=True)["chunks"]

    def transcribe_file_blob(self, audio: str) -> List[Dict[str, Any]]:

        # Decode and write into a virtual file
        decoded = base64.b64decode(audio)
        fileobject = io.BytesIO(decoded)
        filename = str(uuid.uuid4())
        with open(filename, "wb") as f:
            f.write(fileobject.read())

        transcription = self.transcribe_file(filename)

        os.remove(filename)

        return transcription


for model_name in Whisper.configs:
    cfg = Whisper.configs[model_name]
    hub.register(
        model_name,
        TaskType.AUDIO_TRANSCRIPTION,
        Whisper,
        init_args=(model_name,),
        method_name="transcribe_file_blob",
        inputs={"audio": str},  # a numpy array of arbitrary length
        outputs={"text": Batch[str]},
    )
