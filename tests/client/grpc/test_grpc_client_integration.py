import time

import pytest

import nos
from nos.client import InferenceClient
from nos.server import InferenceServiceRuntime
from nos.test.utils import AVAILABLE_RUNTIMES, PyTestGroup, get_benchmark_video


GRPC_PORT = 50055


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
    client = InferenceClient(f"[::]:{GRPC_PORT}")
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
    from itertools import islice

    import cv2

    from nos.common import TaskType, tqdm
    from nos.common.io import VideoReader
    from nos.logging import logger
    from nos.models import StableDiffusion

    # Initialize the server
    container = nos.init(runtime=runtime, port=GRPC_PORT, utilization=1)
    assert container is not None
    assert container.id is not None
    containers = InferenceServiceRuntime.list()
    assert len(containers) == 1

    # Test waiting for server to start
    # This call should be instantaneous as the server is already ready for the test
    client = InferenceClient(f"[::]:{GRPC_PORT}")
    assert client.WaitForServer(timeout=180, retry_interval=5)
    assert client.IsHealthy()

    # Load video for inference
    NOS_TEST_VIDEO = get_benchmark_video()
    video = VideoReader(NOS_TEST_VIDEO)
    assert len(video) > 0
    iterations = 30 if runtime == "cpu" else min(500, len(video))

    # SDv1.4, SDv1.5, SDv2.0, SDv2.1, and SDXL
    task = TaskType.IMAGE_GENERATION
    for _, config in StableDiffusion.configs.items():
        model_name = config.model_name
        model = client.Module(task=task, model_name=model_name)
        assert model is not None
        assert model.GetModelInfo() is not None
        for _ in tqdm(range(1), desc=f"Test [task={task}, model_name={model_name}]"):
            model(prompts=["a cat dancing on the grass."], width=512, height=512, num_images=1)

    # Run object detection over the full video
    logger.info("Running object detection over the full video...")
    video.reset()
    det2d = client.Module(task=TaskType.OBJECT_DETECTION_2D, model_name="yolox/medium")
    for img in tqdm(islice(video, 0, iterations)):
        img = cv2.resize(img, (640, 480))
        predictions = det2d(images=[img])
        assert predictions is not None

    logger.info("Running CLIP over the full video...")
    video.reset()
    clip = client.Module(task=TaskType.IMAGE_EMBEDDING, model_name="openai/clip")
    for img in tqdm(islice(video, 0, iterations)):
        img = cv2.resize(img, (224, 224))
        embeddings = clip(images=[img])
        assert embeddings is not None

    # Shutdown the server
    nos.shutdown()
    containers = InferenceServiceRuntime.list()
    assert len(containers) == 0


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

    from nos.common import TaskType
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
        task=TaskType.IMAGE_GENERATION,
        model_name=f"custom/{model_id}",
        prompts=["a photo of a bench on the moon"],
        width=512,
        height=512,
        num_images=1,
    )
