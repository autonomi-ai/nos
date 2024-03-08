from typing import Dict

import numpy as np
import requests
from PIL import Image

from nos.client import Client


if __name__ == "__main__":
    from typing import List

    client = Client()
    assert client is not None
    assert client.WaitForServer()
    assert client.IsHealthy()

    model_id = "custom/clip-model"
    models: List[str] = client.ListModels()
    assert model_id in models

    model = client.Module(model_id)
    assert model is not None
    assert model.GetModelInfo() is not None
    print(f"Test [model={model_id}]")

    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)

    response: Dict[str, np.ndarray] = model(texts=["two cats sleeping"], images=[image])
    assert isinstance(response, dict)
    assert "image_embeds" in response
    assert "text_embeds" in response
    print(f"image_embeds: {response['image_embeds'].shape}, text_embeds: {response['text_embeds'].shape}")
