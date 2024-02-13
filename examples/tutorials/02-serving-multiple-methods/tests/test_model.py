from typing import Dict, List

import numpy as np
import requests
from PIL import Image

from nos.client import Client


if __name__ == "__main__":
    client = Client()
    assert client is not None
    assert client.WaitForServer()

    model_id = "custom/clip-model-cpu"
    models: List[str] = client.ListModels()
    assert model_id in models

    model = client.Module(model_id)
    assert model is not None
    assert model.GetModelInfo() is not None
    print(f"Test [model={model_id}]")

    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)

    # Test the default __call__ method
    response: Dict[str, np.ndarray] = model(texts=["two cats sleeping"], images=[image])
    assert isinstance(response, dict)
    assert "image_embeds" in response
    assert "text_embeds" in response

    # Test encode_image
    response = model.encode_image(images=[image])
    assert isinstance(response, np.ndarray)
    print(f"image_embeds: {response.shape}")

    # Test encode_text
    response = model.encode_text(texts=["two cats sleeping"])
    assert isinstance(response, np.ndarray)
    print(f"text_embeds: {response.shape}")
