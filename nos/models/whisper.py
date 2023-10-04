from dataclasses import dataclass
from typing import Any, Dict, List

import torch

from nos.hub import HuggingFaceHubConfig


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

    def __init__(self, model_name: str = "openai/whisper-base.en"):
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
