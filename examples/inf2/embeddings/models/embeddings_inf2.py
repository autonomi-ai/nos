"""Embeddings model accelerated with AWS Neuron (using optimum-neuron)."""
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Union

import torch

from nos.constants import NOS_CACHE_DIR
from nos.hub import HuggingFaceHubConfig
from nos.neuron.device import NeuronDevice


@dataclass(frozen=True)
class EmbeddingConfig(HuggingFaceHubConfig):
    """Embeddings model configuration."""

    batch_size: int = 1
    """Batch size for the model."""

    sequence_length: int = 384
    """Sequence length for the model."""


class EmbeddingServiceInf2:
    configs = {
        "BAAI/bge-small-en-v1.5": EmbeddingConfig(
            model_name="BAAI/bge-small-en-v1.5",
        ),
    }

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        from optimum.neuron import NeuronModelForSentenceTransformers
        from transformers import AutoTokenizer

        from nos.logging import logger

        NeuronDevice.setup_environment()
        try:
            self.cfg = EmbeddingServiceInf2.configs[model_name]
        except KeyError:
            raise ValueError(f"Invalid model_name: {model_name}, available models: {self.configs.keys()}")

        # Load model from cache if available, otherwise load from HF and compile
        # (cache is specific to model_name, batch_size and sequence_length)
        cache_dir = (
            NOS_CACHE_DIR / "neuron" / f"{self.cfg.model_name}-bs-{self.cfg.batch_size}-sl-{self.cfg.sequence_length}"
        )
        if Path(cache_dir).exists():
            logger.info(f"Loading model from {cache_dir}")
            self.model = NeuronModelForSentenceTransformers.from_pretrained(str(cache_dir))
            logger.info(f"Loaded model from {cache_dir}")
        else:
            input_shapes = {
                "batch_size": self.cfg.batch_size,
                "sequence_length": self.cfg.sequence_length,
            }
            self.model = NeuronModelForSentenceTransformers.from_pretrained(
                self.cfg.model_name, export=True, **input_shapes
            )
            self.model.save_pretrained(str(cache_dir))
            logger.info(f"Saved model to {cache_dir}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model_name)
        self.logger = logger
        self.logger.info(f"Loaded neuron model: {self.cfg.model_name}")

    @torch.inference_mode()
    def __call__(
        self,
        texts: Union[str, List[str]],
    ) -> Iterable[str]:
        """Embed text with the model."""
        if isinstance(texts, str):
            texts = [texts]
        inputs = self.tokenizer(
            texts,
            padding=True,
            return_tensors="pt",
        )
        outputs = self.model(**inputs)
        return outputs.sentence_embedding.cpu().numpy()
