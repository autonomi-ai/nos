import time

import pytest

import nos
from nos.client import InferenceClient
from nos.server import InferenceServiceRuntime
from nos.test.utils import PyTestGroup, get_benchmark_video


GRPC_PORT = 50055


@pytest.mark.client
@pytest.mark.parametrize("runtime", ["cpu", "gpu", "auto"])
def test_client_init(runtime):  # noqa: F811
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
@pytest.mark.benchmark(group=PyTestGroup.INTEGRATION)
@pytest.mark.parametrize("runtime", ["gpu"])
def test_client_inference_benchmark(runtime):  # noqa: F811
    """Test and benchmark end-to-end client inference interface."""
    from itertools import islice

    import cv2

    from nos.common import TaskType, tqdm
    from nos.common.io import VideoReader
    from nos.logging import logger

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

    # TXT2IMG
    if runtime == "gpu":
        task, model_name = TaskType.IMAGE_GENERATION, "runwayml/stable-diffusion-v1-5"
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
@pytest.mark.parametrize("runtime", ["local"])
def test_client_training(runtime):  # noqa: F811
    """Test end-to-end client training interface."""
    import shutil
    import tempfile
    from pathlib import Path

    from nos.logging import logger
    from nos.test.utils import NOS_TEST_IMAGE

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

    logger.debug("Testing training service...")
    # Copy test image to temporary directory and test training service
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_image = Path(tmp_dir) / "test_image.jpg"
        shutil.copy(NOS_TEST_IMAGE, tmp_image)

        volume_dir = client.Volume("dreambooth_training")
        logger.debug(f"Copying files from {tmp_dir} to {volume_dir}...")
        shutil.copytree(tmp_dir, volume_dir, dirs_exist_ok=True)

        client.Train(
            method="stable-diffusion-dreambooth-lora",
            inputs={
                "model_name": "stabilityai/stable-diffusion-2-1",
                "instance_directory": volume_dir,
                "instance_prompt": "A photo of a bench on the moon",
            },
        )
    logger.debug("Training service test passed.")
