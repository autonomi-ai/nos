import base64
from io import BytesIO
from typing import List, Union

import numpy as np
from PIL import Image
from pydantic import BaseModel


class TextRequest(BaseModel):
    """Text generation request."""

    text: str
    """Text prompt to generate image from."""


class TextResponse(BaseModel):
    """Text generation response."""

    text: str
    """Generated text response."""


class ImageRequest(BaseModel):
    """Image request."""

    image: str
    """Base64 encoded image."""


class ImageResponse(BaseModel):
    """Image response."""

    image: str
    """Base64 encoded image."""

    @classmethod
    def from_pil(cls, image: Image.Image):
        """Helper classmethod to construct ImageResponse from PIL image."""
        file_stream = BytesIO()
        image.save(file_stream, "PNG")
        im_b64 = base64.b64encode(file_stream.getvalue()).decode("utf-8")
        return ImageResponse(image=im_b64)


class VecResponse(BaseModel):
    """Embedding response."""

    embedding: List[float]
    """Embedding vector (fp32)."""

    @classmethod
    def from_numpy(cls, embedding: np.ndarray):
        """Helper classmethod to construct VecResponse from numpy array."""
        return cls(embedding=embedding.tolist())

    def __repr__(self):
        """Concise embedding repr with shape."""
        return f"Embedding ({len(self.embedding)},)"


class Text2ImageRequest(BaseModel):
    """Text-to-Image generation request."""

    prompt: str
    """Text prompt to generate image from."""
    height: int = 512
    """Height of generated image."""
    width: int = 512
    """Width of generated image."""


class Text2ImageResponse(ImageResponse):
    """Text-to-Image generation response."""

    pass


class Text2VecRequest(TextRequest):
    """Text-to-Embedding encoding request."""

    pass


class Text2VecResponse(VecResponse):
    """Text-to-Embedding encoding response."""

    pass


class Image2VecRequest(ImageRequest):
    """Image-to-Embedding encoding request."""

    pass


class Image2VecResponse(VecResponse):
    """Image-to-Embedding encoding response."""

    pass


class PredictionRequest(BaseModel):
    """Generic prediction request for all types of requests."""

    # TODO (spillai): Map this to MethodType enum instead
    method: str
    """Method type."""
    request: Union[Text2ImageRequest, Text2VecRequest, Image2VecRequest, ImageRequest, TextRequest]
    """Request object (txt2img, txt2vec, img2vec)."""


class PredictionResponse(BaseModel):
    """Generic prediction response for all types of responses."""

    response: Union[ImageResponse, VecResponse, TextResponse]
    """Response object (image, embedding, text)."""
