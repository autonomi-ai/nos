from dataclasses import dataclass
from typing import Dict, List, Union

import numpy as np
import torch
from PIL import Image

from nos.common.io import prepare_images
from nos.hub import HuggingFaceHubConfig


@dataclass(frozen=True)
class OwLViTConfig(HuggingFaceHubConfig):
    pass


class OwlViT:
    """OwLViT model for zero-shot text-conditioned object detection."""

    configs = {
        "google/owlv2-large-patch14-ensemble": OwLViTConfig(
            model_name="google/owlv2-large-patch14-ensemble",
        ),
        "google/owlv2-base-patch16-ensemble": OwLViTConfig(
            model_name="google/owlv2-base-patch16-ensemble",
        ),
    }

    def __init__(self, model_name: str = "google/owlv2-base-patch16-ensemble"):
        from transformers import Owlv2ForObjectDetection, Owlv2Processor

        try:
            self.cfg = OwlViT.configs[model_name]
        except KeyError:
            raise ValueError(f"Invalid model_name: {model_name}, available models: {OwlViT.configs.keys()}")
        model_name = self.cfg.model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = Owlv2Processor.from_pretrained(model_name)
        self.model = Owlv2ForObjectDetection.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def __call__(
        self, images: Union[Image.Image, np.ndarray, List[Image.Image], List[np.ndarray]], query: List[str]
    ) -> Dict[str, np.ndarray]:
        """Predict bounding boxes for images conditioned on the query candidate objects.

        Args:
            images (Union[Image.Image, np.ndarray, List[Image.Image], List[np.ndarray]]): Input images.
            query (List[str]): Candidate objects to query (e.g. ["dog", "cat"]).
                `query` is not batched unlike images.
        Returns:
            Dict[str, np.ndarray]: Predicted bounding boxes, scores and labels.
        """
        images = prepare_images(images)
        inputs = self.processor(text=[query], images=images, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            outputs = self.model(**inputs)

            # Target image sizes (height, width) to rescale box predictions [batch_size, 2]
            target_sizes = torch.Tensor([images.size[::-1]]).to(self.device)
            # Convert outputs (bounding boxes and class logits) to COCO API
            (results,) = self.processor.post_process(outputs=outputs, target_sizes=target_sizes)

            # Retrieve predictions for the first image for the corresponding text queries
            boxes, scores, labels = (
                results["boxes"],
                results["scores"],
                results["labels"],
            )
            valid = scores >= self.cfg.score_threshold
            return {
                "bboxes": boxes[valid],
                "scores": scores[valid],
                "labels": labels[valid],
            }
