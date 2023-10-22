from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import torch

from nos import hub
from nos.common import TaskType
from nos.hub import HuggingFaceHubConfig


@dataclass(frozen=True)
class WhisperConfig(HuggingFaceHubConfig):
    pass


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

    def __init__(self, model_name: str = "openai/whisper-tiny.en", dtype: str = "float32"):

        from transformers import pipeline

        try:
            self.cfg = Whisper.configs[model_name]
        except KeyError:
            raise ValueError(f"Invalid model_name: {model_name}, available models: {Whisper.configs.keys()}")

        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")
        self.torch_dtype = getattr(torch, dtype)

        model_name = self.cfg.model_name
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model_name,
            device=self.device,
            torch_dtype=self.torch_dtype,
        )

    def transcribe(
        self, path: Path, chunk_length_s: int = 30, batch_size: int = 1, return_timestamps: bool = True
    ) -> List[Dict[str, Any]]:
        """Transcribe the audio file."""
        with torch.inference_mode():
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
        outputs={"result": List[Dict[str, Any]]},
    )
