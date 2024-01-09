from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import torch

from nos import hub
from nos.common import TaskType
from nos.hub import HuggingFaceHubConfig
from nos.logging import logger


@dataclass(frozen=True)
class WhisperConfig(HuggingFaceHubConfig):
    torch_dtype: str = "float32"


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
        "openai/whisper-large-v3": WhisperConfig(
            model_name="openai/whisper-large-v3",
        ),
        "distil-whisper/distil-small.en": WhisperConfig(
            model_name="distil-whisper/distil-small.en",
            torch_dtype="float16",
        ),
        "distil-whisper/distil-medium.en": WhisperConfig(
            model_name="distil-whisper/distil-medium.en",
            torch_dtype="float16",
        ),
        "distil-whisper/distil-large-v2": WhisperConfig(
            model_name="distil-whisper/distil-large-v2",
            torch_dtype="float16",
        ),
    }

    def __init__(self, model_name: str = "openai/whisper-tiny.en"):

        from transformers import pipeline

        try:
            self.cfg = Whisper.configs[model_name]
        except KeyError:
            raise ValueError(f"Invalid model_name: {model_name}, available models: {Whisper.configs.keys()}")

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        model_name = self.cfg.model_name
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model_name,
            device=self.device,
            torch_dtype=getattr(torch, self.cfg.torch_dtype),
        )

    def transcribe(
        self, path: Path, chunk_length_s: int = 30, batch_size: int = 24, return_timestamps: bool = True
    ) -> List[Dict[str, Any]]:
        """Transcribe the audio file."""
        with torch.inference_mode():
            logger.debug(f"Transcribing audio file [path={path}, size={path.stat().st_size / 1024 / 1024:.2f} MB]")
            # Response is a dictionary with "chunks" and "text" keys
            # We ignore the text key/value since its redundant
            # response: {"chunks": [{'text': ' ...', 'timestamp': (0.0, 5.44)}], "text": " ..."}
            response = self.pipe(
                str(path), chunk_length_s=chunk_length_s, batch_size=batch_size, return_timestamps=return_timestamps
            )
            return response["chunks"]


for model_name in Whisper.configs:
    cfg = Whisper.configs[model_name]
    hub.register(
        model_name,
        TaskType.AUDIO_TRANSCRIPTION,
        Whisper,
        init_args=(model_name,),
        method="transcribe",
        inputs={"path": Path, "chunk_length_s": int, "batch_size": int, "return_timestamps": bool},
        outputs={"chunks": List[Dict[str, Any]]},
    )
