import time
from typing import List

import pytest
from PIL import Image

import nos
from nos.client import Client
from nos.logging import logger
from nos.server import InferenceServiceRuntime
from nos.test.utils import AVAILABLE_RUNTIMES, NOS_TEST_IMAGE, PyTestGroup


GRPC_PORT = 50055

logger.debug(f"AVAILABLE_RUNTIMES={AVAILABLE_RUNTIMES}")


@pytest.mark.client
@pytest.mark.parametrize("runtime", AVAILABLE_RUNTIMES)
def test_grpc_client_init(runtime):  # noqa: F811
    """Test the NOS server daemon initialization."""

    # Initialize the server
    container = nos.init(runtime=runtime, port=GRPC_PORT, utilization=0.5)
    assert container is not None
    assert container.id is not None
    containers = InferenceServiceRuntime.list()
    assert len(containers) == 1

    # Test waiting for server to start
    # This call should be instantaneous as the server is already ready for the test
    client = Client(f"[::]:{GRPC_PORT}")
    assert client.WaitForServer(timeout=180, retry_interval=5)
    assert client.IsHealthy()

    # Test re-initializing the server
    st = time.time()
    container_ = nos.init(runtime=runtime, port=GRPC_PORT, utilization=0.5)
    assert container_.id == container.id
    assert time.time() - st <= 0.5, "Re-initializing the server should be instantaneous, instead took > 0.5s"

    # Shutdown the server
    nos.shutdown()
    containers = InferenceServiceRuntime.list()
    assert len(containers) == 0


@pytest.mark.client
@pytest.mark.parametrize("runtime", ["gpu"])
def test_grpc_client_inference_integration(runtime):  # noqa: F811
    """Test end-to-end client inference interface."""
    from nos.common import ModelSpec, tqdm
    from nos.models import StableDiffusion

    img = Image.open(NOS_TEST_IMAGE)

    # Initialize the server
    container = nos.init(runtime=runtime, port=GRPC_PORT, utilization=1)
    assert container is not None
    assert container.id is not None
    containers = InferenceServiceRuntime.list()
    assert len(containers) == 1

    # Test waiting for server to start
    # This call should be instantaneous as the server is already ready for the test
    client = Client(f"[::]:{GRPC_PORT}")
    assert client.WaitForServer(timeout=180, retry_interval=5)
    assert client.IsHealthy()

    # Get service info
    version = client.GetServiceVersion()
    assert version is not None

    # Check service version
    assert client.CheckCompatibility()

    # List models
    models: List[str] = client.ListModels()
    assert isinstance(models, list)
    assert len(models) >= 1

    # Check GetModelInfo for all models registered
    for model_id in models:
        spec: ModelSpec = client.GetModelInfo(model_id)
        assert spec.task and spec.name
        assert isinstance(spec.signature.inputs, dict)
        assert isinstance(spec.signature.outputs, dict)
        assert len(spec.signature.inputs) >= 1
        assert len(spec.signature.outputs) >= 1

        inputs = spec.signature.get_inputs_spec()
        outputs = spec.signature.get_outputs_spec()
        assert isinstance(inputs, dict)
        assert isinstance(outputs, dict)

    # # TXT2VEC
    # model_id =  "openai/clip-vit-base-patch32"
    # model = client.Module(task=task, model_name=model_name)
    # assert model is not None
    # assert model.GetModelInfo() is not None
    # for _ in tqdm(range(1), desc=f"Test [task={task}, model_name={model_name}]"):
    #     response = model(inputs={"texts": ["a cat dancing on the grass."]})
    #     assert isinstance(response, dict)
    #     assert "embedding" in response

    # IMG2VEC
    model_id = "openai/clip"
    model = client.Module(model_id)
    assert model is not None
    assert model.GetModelInfo() is not None
    for _ in tqdm(range(1), desc=f"Test [model={model_id}]"):
        response = model(images=[img])
        assert isinstance(response, dict)
        assert "embedding" in response

    # IMG2BBOX
    model_id = "yolox/medium"
    model = client.Module("yolox/medium")
    assert model is not None
    assert model.GetModelInfo() is not None
    for _ in tqdm(range(1), desc=f"Test [model={model_id}]"):
        response = model(images=[img])
        assert isinstance(response, dict)

        assert "bboxes" in response

    # TXT2IMG
    # SDv1.4, SDv1.5, SDv2.0, SDv2.1, and SDXL
    for model_id, _config in StableDiffusion.configs.items():
        model = client.Module(model_id)
        assert model is not None
        spec: ModelSpec = model.GetModelInfo()
        assert spec is not None
        assert isinstance(spec, ModelSpec)

        for _ in tqdm(range(1), desc=f"Test [model={model_id}]"):
            model(prompts=["a cat dancing on the grass."], width=512, height=512, num_images=1)


@pytest.mark.skip(reason="Fine-tuning is not supported yet.")
@pytest.mark.client
@pytest.mark.benchmark(group=PyTestGroup.INTEGRATION)
@pytest.mark.parametrize(
    "client_with_server",
    ("local_grpc_client_with_server",),
)
def test_grpc_client_training(client_with_server, request):  # noqa: F811
    """Test end-to-end client training interface."""
    import shutil
    from pathlib import Path

    from nos.logging import logger
    from nos.test.utils import NOS_TEST_IMAGE

    # Test waiting for server to start
    # This call should be instantaneous as the server is already ready for the test
    client = request.getfixturevalue(client_with_server)
    assert client.IsHealthy()

    # Create a temporary volume for training images
    volume_dir = client.Volume("dreambooth_training")

    logger.debug("Testing training service...")

    # Copy test image to volume and test training service
    tmp_image = Path(volume_dir) / "test_image.jpg"
    shutil.copy(NOS_TEST_IMAGE, tmp_image)

    # Train a new LoRA model with the image of a bench
    response = client.Train(
        method="stable-diffusion-dreambooth-lora",
        inputs={
            "model_name": "stabilityai/stable-diffusion-2-1",
            "instance_directory": volume_dir,
            "instance_prompt": "A photo of a bench on the moon",
            "max_train_steps": 10,
        },
    )
    assert response is not None
    model_id = response["job_id"]
    logger.debug(f"Training service test completed [model_id={model_id}].")

    # Wait for the model to be ready
    # For e.g. model_id = "stable-diffusion-dreambooth-lora_16cd4490"
    # model_id = "stable-diffusion-dreambooth-lora_ef939db5"
    response = client.Wait(job_id=model_id, timeout=600, retry_interval=5)
    logger.debug(f"Training service test completed [model_id={model_id}, response={response}].")
    time.sleep(10)

    # Test inference with the trained model
    logger.debug("Testing inference service...")
    response = client.Run(
        f"custom/{model_id}",
        inputs={"prompts": ["a photo of a bench on the moon"], "width": 512, "height": 512, "num_images": 1},
    )
