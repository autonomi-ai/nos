"""Fully integrated gRPC client test with gRPC-based inference server.

The test spins up a gRPC inferennce server and then sends requests to it using the gRPC client.

"""

import numpy as np
import pytest
from PIL import Image
from tqdm import tqdm

from nos.test.utils import NOS_TEST_IMAGE


pytestmark = pytest.mark.e2e


def test_e2e_grpc_client_and_server(test_grpc_server, test_grpc_client):
    img = Image.open(NOS_TEST_IMAGE)
    img = np.array(img)

    # List models
    models = test_grpc_client.ListModels()
    assert isinstance(models, list)
    assert len(models) >= 1

    # TXT2VEC
    method, model_name = "txt2vec", "openai/clip-vit-base-patch32"
    for _ in tqdm(range(10), desc=f"Bench [method{method}, model_name={model_name}]"):
        response = test_grpc_client.Predict(method=method, model_name=model_name, text="a cat dancing on the grass.")
        assert isinstance(response, dict)
        assert "embedding" in response

    # IMG2VEC
    method, model_name = "img2vec", "openai/clip-vit-base-patch32"
    for _ in tqdm(range(10), desc=f"Bench [method{method}, model_name={model_name}]"):
        response = test_grpc_client.Predict(method=method, model_name=model_name, img=img)
        assert isinstance(response, dict)
        assert "embedding" in response

    # TXT2IMG
    method, model_name = "txt2img", "stabilityai/stable-diffusion-2"
    for _ in tqdm(range(2), desc=f"Bench [method{method}, model_name={model_name}]"):
        response = test_grpc_client.Predict(method=method, model_name=model_name, text="a cat dancing on the grass.")
        assert isinstance(response, dict)
        assert "image" in response
