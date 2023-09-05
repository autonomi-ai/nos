from dataclasses import dataclass, field
from typing import Any, Dict, List

import torch

from nos.common.spec import RuntimeEnv
from nos.hub import TorchHubConfig


GIT_TAG = "v3.1.1"


@dataclass(frozen=True)
class WhisperXConfig(TorchHubConfig):
    """WhisperX model configuration."""

    chunk_length_s: int = 30
    """Chunk length in seconds."""

    runtime_env: RuntimeEnv = field(
        init=False,
        default_factory=lambda: RuntimeEnv(
            conda={
                "dependencies": [
                    "pip",
                    {
                        "pip": [
                            "torch==2.0",
                            "torchaudio==2.0.0",
                            f"https://github.com/m-bain/whisperX/archive/refs/tags/${GIT_TAG}.zip",
                        ]
                    },
                ]
            }
        ),
    )
    """Runtime environment specification for WhisperX."""


class WhisperX:
    """WhisperX model for audio transcription.
    https://github.com/m-bain/whisperX
    """

    configs = {
        "m-bain/whisperx-large": WhisperXConfig(
            repo="m-bain/whisperX",
            model_name="large",
        ),
    }

    def __init__(self, model_name: str = "m-bain/whisperx-large"):
        import whisperx

        try:
            self.cfg = WhisperXConfig.configs[model_name]
        except KeyError:
            raise ValueError(f"Invalid model_name: {model_name}, available models: {WhisperXConfig.configs.keys()}")

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            torch.backends.cuda.matmul.allow_tf32 = True
        else:
            raise RuntimeError("No CUDA device available")

        self.model = whisperx.load_model(self.cfg.model_name, self.device)
        self.load_align_model = whisperx.load_align_model
        self.align = whisperx.align

    def transcribe_file(self, filename: str) -> List[Dict[str, Any]]:
        """Transcribe the audio file."""
        with torch.inference_mode():
            # transcribe with original whisper
            result = self.model.transcribe(filename)

            # load alignment model and metadata
            model_a, metadata = self.load_align_model(language_code=result["language"], device=self.device)

            # align whisper output
            # result_aligned = [{...}]
            result_aligned = self.align(result["segments"], model_a, metadata, filename, self.device)
            return result_aligned
