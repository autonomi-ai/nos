from dataclasses import dataclass
from typing import Any, List, Union

import torch

from nos.hub import HuggingFaceHubConfig


@dataclass(frozen=True)
class TextToSpeechConfig(HuggingFaceHubConfig):
    pass


class TextToSpeech:
    """Text-to-speech models for speech generation.
    https://huggingface.co/tasks/text-to-speech
    """

    configs = {
        "suno/bark": TextToSpeechConfig(
            model_name="suno/bark",
        ),
        "suno/bark-small": TextToSpeechConfig(
            model_name="suno/bark-small",
        ),
    }

    def __init__(self, model_name: str = "suno/bark-small"):
        from transformers import AutoModel, AutoProcessor

        try:
            self.cfg = TextToSpeech.configs[model_name]
        except KeyError:
            raise ValueError(f"Invalid model_name: {model_name}, available models: {TextToSpeech.configs.keys()}")

        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            torch.backends.cuda.matmul.allow_tf32 = True
        else:
            self.device = torch.device("cpu")

        model_name = self.cfg.model_name
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)

    def __call__(self, prompts: Union[str, List[str]]) -> Any:
        """Generate speech from text prompts."""
        with torch.inference_mode():
            inputs = self.processor(
                text=prompts,
                return_tensors="pt",
            ).to(self.device)
            return self.model.generate(**inputs, do_sample=True)
