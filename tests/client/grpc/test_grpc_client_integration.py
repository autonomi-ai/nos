import time
from pathlib import Path
from typing import Iterable, List

import pytest
from PIL import Image

from nos.logging import logger
from nos.test.conftest import (
    GRPC_CLIENT_SERVER_CONFIGURATIONS,
)
from nos.test.utils import NOS_TEST_AUDIO, NOS_TEST_IMAGE, PyTestGroup


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

    _test_grpc_client_inference_noop(client)
    _test_grpc_client_inference_models(client)


@pytest.mark.parametrize("client_with_server", GRPC_CLIENT_SERVER_CONFIGURATIONS)
def test_grpc_client_inference(client_with_server, request):  # noqa: F811
    """Test end-to-end client inference interface (pytest fixtures + client-server integration tests).

    This test spins up a gRPC inference server within a
    GPU docker-runtime environment initialized and then issues
    requests to it using the gRPC client.
    """

    client = request.getfixturevalue(client_with_server)
    assert client is not None
    assert client.IsHealthy()

    _test_grpc_client_inference_spec(client)
    _test_grpc_client_inference_noop(client)
    _test_grpc_client_inference_models(client)


def _test_grpc_client_inference_spec(client):  # noqa: F811
    from nos.common import ImageSpec, ModelSpec, ModelSpecMetadataCatalog, ObjectTypeInfo, TaskType, TensorSpec

    # List models
    models: List[str] = client.ListModels()
    assert isinstance(models, list)
    assert len(models) >= 1

    # Check GetModelInfo for all models registered
    for model_id in models:
        spec: ModelSpec = client.GetModelInfo(model_id)

        # Tests the internal model catalog serialization to ensure
        # that the metadata, profile and resource catalogs are relayed
        # to the client correctly.
        catalog: ModelSpecMetadataCatalog = client._get_model_catalog()
        assert catalog is not None
        assert isinstance(catalog, ModelSpecMetadataCatalog)

        # Check that the model spec is valid
        assert spec.task() and spec.name
        assert spec.signature is not None
        assert len(spec.signature) > 0
        assert isinstance(spec.default_signature.parameters, dict)
        assert len(spec.default_signature.parameters) >= 1
        assert spec.default_signature.return_annotation is not None

        inputs = spec.default_signature.get_inputs_spec()
        outputs = spec.default_signature.get_outputs_spec()
        assert isinstance(inputs, dict)
        for _, v in inputs.items():
            assert isinstance(v, (list, ObjectTypeInfo))
            if isinstance(v, list):
                assert isinstance(v[0], ObjectTypeInfo)
        assert isinstance(outputs, (dict, ObjectTypeInfo))

        for method in spec.signature:
            task: TaskType = spec.task(method)
            assert isinstance(task, TaskType)
            assert task.value is not None
            logger.debug(f"Testing model [id={model_id}, spec={spec}, method={method}, task={spec.task(method)}]")
            inputs = spec.signature[method].get_inputs_spec()
            outputs = spec.signature[method].get_outputs_spec()
            assert isinstance(inputs, dict)
            for _, v in inputs.items():
                assert isinstance(v, (list, ObjectTypeInfo))
                if isinstance(v, list):
                    assert isinstance(v[0], ObjectTypeInfo)
            assert isinstance(outputs, (dict, ObjectTypeInfo))

            for _, type_info in inputs.items():
                assert isinstance(type_info, (list, ObjectTypeInfo))
                if isinstance(type_info, ObjectTypeInfo):
                    assert type_info.base_spec() is None or isinstance(type_info.base_spec(), (ImageSpec, TensorSpec))
                    assert type_info.base_type() is not None


def _test_grpc_client_inference_noop(client):  # noqa: F811
    from multiprocessing.pool import ThreadPool

    from nos.common import tqdm

    # Get service info
    version = client.GetServiceVersion()
    assert version is not None

    # Check service version
    assert client.CheckCompatibility()

    # List models
    models: List[str] = client.ListModels()
    assert isinstance(models, list)
    assert len(models) >= 1

    # Test UploadFile
    assert isinstance(NOS_TEST_IMAGE, Path)
    with client.UploadFile(NOS_TEST_IMAGE) as remote_path:
        assert client.Run("noop/process-file", inputs={"path": remote_path})

    # noop/process-images with default method
    img = Image.open(NOS_TEST_IMAGE).resize((224, 224))
    response = client.Run("noop/process-images", inputs={"images": [img]})
    assert isinstance(response, list)

    # noop/process-texts with default method
    response = client.Run("noop/process-texts", inputs={"texts": ["a cat dancing on the grass"]})
    assert isinstance(response, list)

    # noop/process
    model_id = "noop/process"
    model = client.Module(model_id)
    assert model is not None
    assert model.GetModelInfo() is not None
    for _ in tqdm(range(1), desc=f"Test [model={model_id}]"):
        response = client.Run(model_id, inputs={"images": [img]}, method="process_images")
        assert isinstance(response, list)

        response = client.Run(model_id, inputs={"texts": ["a cat dancing on the grass"]}, method="process_texts")
        assert isinstance(response, list)

        response = model.process_images(images=[img])
        assert isinstance(response, list)

        response = model.process_texts(texts=["a cat dancing on the grass."])
        assert isinstance(response, list)

        idx = 0
        texts = ["a cat dancing on the grass.", "a dog running on the beach."]
        response: Iterable[str] = client.Stream(model_id, inputs={"texts": texts}, method="stream_texts")
        for resp in response:
            assert resp is not None
            assert isinstance(resp, str)
            idx += 1
        assert idx > len(texts)

        idx = 0
        response: Iterable[str] = model.stream_texts(texts=texts, _stream=True)
        for resp in response:
            assert resp is not None
            assert isinstance(resp, str)
            idx += 1
        assert idx > len(texts)

    # noop/process scaling
    model_id = "noop/process"
    model = client.Module(model_id)

    num_replicas, num_iters = 2, 5
    model.Load(num_replicas=num_replicas)

    # Spin up 2 replicas to execute 5 inferences each
    st = time.time()
    with ThreadPool(processes=num_replicas) as pool:
        # Execute 5 inferences per thread
        responses = []
        for _ in tqdm(range(num_replicas * num_iters), desc=f"Test [model={model_id}]"):
            response = pool.apply_async(
                func=model.process_sleep,
                kwds={"seconds": 1.0},
            )
            responses.append(response)

        # Wait for all threads to complete
        for response in responses:
            assert response.get()
            assert isinstance(response.get(), bool)
    end = time.time()

    # Check that the total time taken is less than some
    # fixed overhead (2 seconds) + 5 iterations * 1.2 (20% overhead)
    total_time = end - st
    assert total_time < 2 + num_iters * 1.2
    logger.debug(f"Total time taken for {num_replicas} replicas: {total_time:.2f} seconds.")


def _test_grpc_client_inference_models(client):  # noqa: F811
    from nos.common import ModelSpec, tqdm

    img = Image.open(NOS_TEST_IMAGE).resize((224, 224))

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

    # Whisper
    model_id = "openai/whisper-small.en"
    model = client.Module(model_id)
    assert model is not None
    assert model.GetModelInfo() is not None
    for _ in tqdm(range(1), desc=f"Test [model={model_id}]"):
        # Uplaod local audio path to server and execute inference
        # on the remote path. Note that the audio file is deleted
        # from the server after the inference is complete via the
        # context manager.
        logger.debug(
            f"Uploading audio file to server [path={NOS_TEST_AUDIO}, size={Path(NOS_TEST_AUDIO).stat().st_size / 1024 / 1024:.2f} MB]"
        )
        with client.UploadFile(NOS_TEST_AUDIO) as remote_path:
            assert isinstance(remote_path, Path)
            response = model.transcribe(path=remote_path)
            assert isinstance(response, dict)
            assert "chunks" in response
            for item in response["chunks"]:
                assert "timestamp" in item
                assert "text" in item

    # TXT2IMG
    # SDv2.1, and SDXL
    for model_id in ("stabilityai/stable-diffusion-2-1",):
        model = client.Module(model_id)
        assert model is not None
        spec: ModelSpec = model.GetModelInfo()
        assert spec is not None
        assert isinstance(spec, ModelSpec)
        for _ in tqdm(range(1), desc=f"Test [model={model_id}]"):
            model(prompts=["a cat dancing on the grass."], width=512, height=512, num_images=1, num_inference_steps=10)


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
