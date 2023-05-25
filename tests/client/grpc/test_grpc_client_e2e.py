"""Fully integrated gRPC client test with gRPC-based inference server.

The test spins up a gRPC inferennce server and then sends requests to it using the gRPC client.

"""

import numpy as np
import pytest
from PIL import Image
from tqdm import tqdm


pytestmark = pytest.mark.client

from nos.test.utils import NOS_TEST_IMAGE  # noqa: E402


@pytest.mark.client
def test_e2e_grpc_client_and_gpu_server(grpc_client_with_gpu_backend):  # noqa: F811
    """Test the gRPC client with GPU docker runtime initialized.

    This test spins up a gRPC inference server within a
    GPU docker-runtime environment initialized and then issues
    requests to it using the gRPC client.
    """
    client = grpc_client_with_gpu_backend

    img = Image.open(NOS_TEST_IMAGE)
    img = np.array(img)

    # Test client health
    assert client.IsHealthy()

    # Test waiting for server to start
    # This call should be instantaneous as the server is already ready for the test
    assert client.WaitForServer(timeout=180, retry_interval=5)

    # Get service info
    version = client.GetServiceVersion()
    assert version is not None

    # List models
    models = client.ListModels()
    assert isinstance(models, list)
    assert len(models) >= 1

    # Check if GetModelInfo raises error
    model_name = "openai/clip-vit-base-patch32"
    with pytest.raises(Exception):
        client.GetModelInfo(model_name=model_name)

    # TXT2VEC
    method, model_name = "txt2vec", "openai/clip-vit-base-patch32"
    for _ in tqdm(range(1), desc=f"Bench [method{method}, model_name={model_name}]"):
        response = client.Predict(method=method, model_name=model_name, text="a cat dancing on the grass.")
        assert isinstance(response, dict)
        assert "embedding" in response

    # IMG2VEC
    method, model_name = "img2vec", "openai/clip-vit-base-patch32"
    for _ in tqdm(range(1), desc=f"Bench [method{method}, model_name={model_name}]"):
        response = client.Predict(method=method, model_name=model_name, img=img)
        assert isinstance(response, dict)
        assert "embedding" in response

    # TXT2IMG
    method, model_name = "txt2img", "stabilityai/stable-diffusion-2"
    for _ in tqdm(range(1), desc=f"Bench [method{method}, model_name={model_name}]"):
        response = client.Predict(method=method, model_name=model_name, text="a cat dancing on the grass.")
        assert isinstance(response, dict)
        assert "image" in response

    # IMG2BBOX
    method, model_name = "img2bbox", "open-mmlab/faster-rcnn"
    for _ in tqdm(range(1), desc=f"Bench [method{method}, model_name={model_name}]"):
        response = client.Predict(method=method, model_name=model_name, img=img)
        assert isinstance(response, dict)


@pytest.mark.client
def test_e2e_grpc_client_and_cpu_server(grpc_client_with_cpu_backend):  # noqa: F811
    """Test the gRPC client with CPU docker runtime initialized.

    This test spins up a gRPC inference server within a
    CPU docker-runtime environment initialized and then issues
    requests to it using the gRPC client.
    """
    client = grpc_client_with_cpu_backend

    img = Image.open(NOS_TEST_IMAGE)
    img = np.array(img)

    # Test client health
    assert client.IsHealthy()

    # Test waiting for server to start
    # This call should be instantaneous as the server is already ready for the test
    assert client.WaitForServer(timeout=180, retry_interval=5)

    # List models
    models = client.ListModels()
    assert isinstance(models, list)
    assert len(models) >= 1

    # Check if GetModelInfo raises error
    model_name = "openai/clip-vit-base-patch32"
    with pytest.raises(Exception):
        client.GetModelInfo(model_name=model_name)

    # TXT2VEC
    method, model_name = "txt2vec", "openai/clip-vit-base-patch32"
    for _ in tqdm(range(1), desc=f"Bench [method{method}, model_name={model_name}]"):
        response = client.Predict(method=method, model_name=model_name, text="a cat dancing on the grass.")
        assert isinstance(response, dict)
        assert "embedding" in response

    # IMG2VEC
    method, model_name = "img2vec", "openai/clip-vit-base-patch32"
    for _ in tqdm(range(1), desc=f"Bench [method{method}, model_name={model_name}]"):
        response = client.Predict(method=method, model_name=model_name, img=img)
        assert isinstance(response, dict)
        assert "embedding" in response

    # TXT2IMG: CPU backend does not support txt2img and should raise an exception
    method, model_name = "txt2img", "stabilityai/stable-diffusion-2"
    for _ in tqdm(range(1), desc=f"Bench [method{method}, model_name={model_name}]"):
        with pytest.raises(Exception):
            response = client.Predict(method=method, model_name=model_name, text="a cat dancing on the grass.")

    # IMG2BBOX
    method, model_name = "img2bbox", "open-mmlab/faster-rcnn"
    for _ in tqdm(range(1), desc=f"Bench [method{method}, model_name={model_name}]"):
        with pytest.raises(Exception):
            response = client.Predict(method=method, model_name=model_name, img=img)
