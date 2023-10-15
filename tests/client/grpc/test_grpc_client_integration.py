import time
from typing import List

import pytest
from PIL import Image

from nos.logging import logger
from nos.test.conftest import (
    CLIENT_SERVER_CONFIGURATIONS,
)
from nos.test.utils import NOS_TEST_IMAGE, PyTestGroup


INTEGRATION_TEST_RUNTIMES = ["cpu", "gpu"]
logger.debug(f"INTEGRATION TEST RUNTIMES={INTEGRATION_TEST_RUNTIMES}")


pytestmark = pytest.mark.client


@pytest.mark.skip(reason="Not implemented yet.")
@pytest.mark.parametrize("runtime", INTEGRATION_TEST_RUNTIMES)
def test_grpc_client_inference_integration(runtime, request):  # noqa: F811
    """Test end-to-end client inference interface (nos.init() + client-server integration tests)."""

    client = request.getfixturevalue(f"grpc_client_with_{runtime}_backend")
    assert client is not None
    assert client.IsHealthy()

    _test_grpc_client_inference(client)


@pytest.mark.parametrize("client_with_server", CLIENT_SERVER_CONFIGURATIONS)
def test_grpc_client_inference(client_with_server, request):  # noqa: F811
    """Test end-to-end client inference interface (pytest fixtures + client-server integration tests).

    This test spins up a gRPC inference server within a
    GPU docker-runtime environment initialized and then issues
    requests to it using the gRPC client.
    """

    client = request.getfixturevalue(client_with_server)
    assert client is not None
    assert client.IsHealthy()

    _test_grpc_client_inference(client)


def _test_grpc_client_inference(client):  # noqa: F811
    from nos.common import ImageSpec, ModelSpec, ObjectTypeInfo, TaskType, TensorSpec, tqdm

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
        assert spec.task() and spec.name
        assert spec.signature is not None
        assert len(spec.signature) > 0
        assert isinstance(spec.default_signature.inputs, dict)
        assert isinstance(spec.default_signature.outputs, dict)
        assert len(spec.default_signature.inputs) >= 1
        assert len(spec.default_signature.outputs) >= 1

        inputs = spec.default_signature.get_inputs_spec()
        outputs = spec.default_signature.get_outputs_spec()
        assert isinstance(inputs, dict)
        assert isinstance(outputs, dict)

        for method in spec.signature:
            task: TaskType = spec.task(method)
            assert isinstance(task, TaskType)
            assert task.value is not None
            logger.debug(f"Testing model [id={model_id}, spec={spec}, method={method}, task={spec.task(method)}]")
            inputs = spec.signature[method].get_inputs_spec()
            outputs = spec.signature[method].get_outputs_spec()
            assert isinstance(inputs, dict)
            assert isinstance(outputs, dict)

            for _, type_info in inputs.items():
                assert isinstance(type_info, (list, ObjectTypeInfo))
                if isinstance(type_info, ObjectTypeInfo):
                    assert type_info.base_spec() is None or isinstance(type_info.base_spec(), (ImageSpec, TensorSpec))
                    assert type_info.base_type() is not None

    # noop/process-images with default method
    img = Image.open(NOS_TEST_IMAGE)
    response = client.Run("noop/process-images", inputs={"images": [img]})
    assert isinstance(response, dict)
    assert "result" in response

    # noop/process-texts with default method
    response = client.Run("noop/process-texts", inputs={"texts": ["a cat dancing on the grass"]})
    assert isinstance(response, dict)
    assert "result" in response

    # noop/process
    model_id = "noop/process"
    model = client.Module(model_id)
    assert model is not None
    assert model.GetModelInfo() is not None
    for _ in tqdm(range(1), desc=f"Test [model={model_id}]"):
        response = client.Run(model_id, inputs={"images": [img]}, method="process_images")
        assert isinstance(response, dict)
        assert "result" in response

        response = client.Run(model_id, inputs={"texts": ["a cat dancing on the grass"]}, method="process_texts")
        assert isinstance(response, dict)
        assert "result" in response

        response = model.process_images(images=[img])
        assert isinstance(response, dict)
        assert "result" in response

        response = model.process_texts(texts=["a cat dancing on the grass."])
        assert isinstance(response, dict)
        assert "result" in response

    # TXT2VEC / IMG2VEC
    model_id = "openai/clip"
    model = client.Module(model_id)
    assert model is not None
    assert model.GetModelInfo() is not None
    # TXT2VEC
    for _ in tqdm(range(1), desc=f"Test [model={model_id}]"):
        response = model.encode_text(texts=["a cat dancing on the grass."])
        assert isinstance(response, dict)
        assert "embedding" in response

        # explicit call to encode_text
        response = model(texts=["a cat dancing on the grass"], _method="encode_text")
        assert isinstance(response, dict)
        assert "embedding" in response
    # IMG2VEC
    for _ in tqdm(range(1), desc=f"Test [model={model_id}]"):
        # explicit call to encode_image
        response = model(images=[img])
        assert isinstance(response, dict)
        assert "embedding" in response

        # explicit call to encode_image
        response = model(images=[img], _method="encode_image")
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
    from nos.models import StableDiffusion

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
