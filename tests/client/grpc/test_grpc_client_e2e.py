"""Fully integrated gRPC client test with gRPC-based inference server.

The test spins up a gRPC inferennce server and then sends requests to it using the gRPC client.

"""

import numpy as np
import pytest
from PIL import Image
from tqdm import tqdm


pytestmark = pytest.mark.client

from nos.common import TaskType  # noqa: E402
from nos.test.utils import NOS_TEST_IMAGE  # noqa: E402


@pytest.mark.skip(reason="This test is not ready yet.")
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

    # Check service version
    assert client.CheckCompatibility()

    # List models
    models = client.ListModels()
    assert isinstance(models, list)
    assert len(models) >= 1

    # Check GetModelInfo for all models registered
    for spec in models:
        spec = client.GetModelInfo(spec)
        assert spec.task and spec.name
        assert isinstance(spec.signature.inputs, dict)
        assert isinstance(spec.signature.outputs, dict)
        assert len(spec.signature.inputs) >= 1
        assert len(spec.signature.outputs) >= 1

    # TXT2VEC
    task, model_name = TaskType.TEXT_EMBEDDING, "openai/clip-vit-base-patch32"
    model = client.Module(task=task, model_name=model_name)
    assert model is not None
    assert model.GetModelInfo() is not None
    for _ in tqdm(range(1), desc=f"Bench [task={task}, model_name={model_name}]"):
        response = model(text="a cat dancing on the grass.")
        assert isinstance(response, dict)
        assert "embedding" in response

    # IMG2VEC
    task, model_name = TaskType.IMAGE_EMBEDDING, "openai/clip-vit-base-patch32"
    model = client.Module(task=task, model_name=model_name)
    assert model is not None
    assert model.GetModelInfo() is not None
    for _ in tqdm(range(1), desc=f"Bench [task={task}, model_name={model_name}]"):
        response = model(img=img)
        assert isinstance(response, dict)
        assert "embedding" in response

    # IMG2BBOX
    task, model_name = TaskType.OBJECT_DETECTION_2D, "torchvision/fasterrcnn_mobilenet_v3_large_320_fpn"
    model = client.Module(task=task, model_name=model_name)
    assert model is not None
    assert model.GetModelInfo() is not None
    for _ in tqdm(range(1), desc=f"Bench [task={task}, model_name={model_name}]"):
        response = model(img=img)
        assert isinstance(response, dict)

        assert "bboxes" in response

    # TXT2IMG
    task, model_name = TaskType.IMAGE_GENERATION, "stabilityai/stable-diffusion-2"
    model = client.Module(task=task, model_name=model_name)
    assert model is not None
    assert model.GetModelInfo() is not None
    for _ in tqdm(range(1), desc=f"Bench [task={task}, model_name={model_name}]"):
        response = model(text="a cat dancing on the grass.")
        assert isinstance(response, dict)
        assert "image" in response


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

    # Check service version
    assert client.CheckCompatibility()

    # List models
    models = client.ListModels()
    assert isinstance(models, list)
    assert len(models) >= 1

    # Check GetModelInfo for all models registered
    for spec in models:
        spec = client.GetModelInfo(spec)
        assert spec.task and spec.name
        assert isinstance(spec.signature.inputs, dict)
        assert isinstance(spec.signature.outputs, dict)
        assert len(spec.signature.inputs) >= 1
        assert len(spec.signature.outputs) >= 1

    # TXT2VEC
    task, model_name = TaskType.TEXT_EMBEDDING, "openai/clip-vit-base-patch32"
    model = client.Module(task=task, model_name=model_name)
    assert model is not None
    assert model.GetModelInfo() is not None
    for _ in tqdm(range(1), desc=f"Bench [task={task}, model_name={model_name}]"):
        response = model(texts="a cat dancing on the grass.")
        assert isinstance(response, dict)
        assert "embedding" in response

    # IMG2VEC
    task, model_name = TaskType.IMAGE_EMBEDDING, "openai/clip-vit-base-patch32"
    model = client.Module(task=task, model_name=model_name)
    assert model is not None
    assert model.GetModelInfo() is not None
    for _ in tqdm(range(1), desc=f"Bench [task={task}, model_name={model_name}]"):
        response = model(images=img)
        assert isinstance(response, dict)
        assert "embedding" in response

    # # IMG2BBOX
    # task, model_name = TaskType.OBJECT_DETECTION_2D, "torchvision/fasterrcnn_mobilenet_v3_large_320_fpn"
    # model = client.Module(task=task, model_name=model_name)
    # assert model is not None
    # assert model.GetModelInfo() is not None
    # for _ in tqdm(range(1), desc=f"Bench [task={task}, model_name={model_name}]"):
    #     response = model(img=img)
    #     assert isinstance(response, dict)
    #     assert "bboxes" in response

    # TXT2IMG (not supported in CPU yet)
    task, model_name = TaskType.IMAGE_GENERATION, "stabilityai/stable-diffusion-2"
    model = client.Module(task=task, model_name=model_name)
    assert model is not None
    assert model.GetModelInfo() is not None
    for _ in tqdm(range(1), desc=f"Bench [task={task}, model_name={model_name}]"):
        with pytest.raises(Exception):
            response = model(text="a cat dancing on the grass.")
