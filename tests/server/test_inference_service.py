import numpy as np
import pytest
from PIL import Image
from tqdm import tqdm

from nos.common import TaskType  # noqa: E402
from nos.test.utils import NOS_TEST_IMAGE  # noqa: E402


pytestmark = pytest.mark.server


def test_inference_service_impl(grpc_client_with_server):  # noqa: F811
    client = grpc_client_with_server
    assert client is not None
    assert client.IsHealthy()


@pytest.mark.benchmark
@pytest.mark.parametrize(
    "model_name",
    [
        "openai/clip-vit-base-patch32",
    ],
)
@pytest.mark.parametrize("scale", [1, 8])
def test_inference_service_shm_transport(model_name, scale, grpc_client_with_server):  # noqa: F811
    """Benchmark shared memory transport and inference between the client-server.

    Note: This test is only valid for the local server.
    """
    import time

    client = grpc_client_with_server
    assert client is not None

    img = Image.open(NOS_TEST_IMAGE)
    W, H = 224, 224
    img = img.resize((W * scale, H * scale))
    img = np.asarray(img)

    # Load model
    task = TaskType.IMAGE_EMBEDDING
    model = client.Module(task=task, model_name=model_name)
    assert model is not None
    assert model.GetModelInfo() is not None

    # Warmup
    inputs_shm = None
    for _ in tqdm(range(10), desc=f"Warmup [task={task}, model_name={model_name}]", total=0):
        inputs = {"images": [img]}
        if inputs_shm is None:
            inputs_shm = inputs
            model.RegisterSystemSharedMemory(inputs)
        response = model(**inputs)
        assert isinstance(response, dict)
        assert "embedding" in response
        assert isinstance(response["embedding"], np.ndarray)
    model.UnregisterSystemSharedMemory()

    # Benchmark (10s)
    for b in range(0, 8):
        B = 2**b
        images = np.stack([img for _ in range(B)])
        st = time.time()
        inputs_shm = None
        for _ in tqdm(
            range(100_000),
            desc=f"Benchmark model={model_name}, task={task} [B={B}, shape={img.shape}]",
            unit="images",
            unit_scale=B,
            total=0,
        ):
            inputs = {"images": images}
            if inputs_shm is None:
                inputs_shm = inputs
                model.RegisterSystemSharedMemory(inputs)
            response = model(**inputs)
            assert isinstance(response, dict)
            assert "embedding" in response
            assert isinstance(response["embedding"], np.ndarray)
            N, _ = response["embedding"].shape
            assert N == B
            if time.time() - st > 10.0:
                break
        model.UnregisterSystemSharedMemory()
