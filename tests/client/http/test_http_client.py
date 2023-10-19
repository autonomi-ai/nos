import pytest

from nos.test.conftest import (  # noqa: F401, E402
    HTTP_CLIENT_SERVER_CONFIGURATIONS,
    http_client_with_cpu_backend,
    http_client_with_gpu_backend,
    local_http_client_with_server,
)
from nos.test.utils import NOS_TEST_IMAGE


pytestmark = pytest.mark.client


@pytest.mark.parametrize("client_with_server", HTTP_CLIENT_SERVER_CONFIGURATIONS)
def test_http_client(client_with_server, request):  # noqa: F811
    http_client = request.getfixturevalue(client_with_server)
    assert http_client is not None

    # Health check
    response = http_client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.parametrize("client_with_server", HTTP_CLIENT_SERVER_CONFIGURATIONS)
def test_http_client_inference_object_detection_2d(client_with_server, request):
    http_client = request.getfixturevalue(client_with_server)
    assert http_client is not None

    import numpy as np
    from PIL import Image

    from nos.server.http._utils import encode_dict

    # Test inference with JSON encoding
    data = {
        "model_id": "yolox/small",
        "inputs": {
            "images": Image.open(NOS_TEST_IMAGE),
        },
    }
    response = http_client.post(
        "/infer",
        headers={"Content-Type": "application/json"},
        json=encode_dict(data),
    )
    assert response.status_code == 201, response.text
    predictions = response.json()
    assert isinstance(predictions, dict)

    assert "labels" in predictions
    (B, N) = np.array(predictions["labels"]).shape
    assert B == 1

    assert "scores" in predictions
    assert np.array(predictions["scores"]).shape[-1] == N

    assert "bboxes" in predictions
    assert np.array(predictions["bboxes"]).shape[-2:] == (N, 4)


@pytest.mark.parametrize("client_with_server", HTTP_CLIENT_SERVER_CONFIGURATIONS)
def test_http_client_inference_clip_embedding(client_with_server, request):
    http_client = request.getfixturevalue(client_with_server)
    assert http_client is not None

    import numpy as np
    from PIL import Image

    from nos.server.http._utils import encode_dict

    # Test inference with JSON encoding
    model_id = "openai/clip"
    data = {
        "model_id": model_id,
        "method": "encode_image",
        "inputs": {
            "images": Image.open(NOS_TEST_IMAGE),
        },
    }
    response = http_client.post(
        "/infer",
        headers={"Content-Type": "application/json"},
        json=encode_dict(data),
    )
    assert response.status_code == 201, response.text
    result = response.json()
    assert isinstance(result, dict)

    assert "embedding" in result
    (B, N) = np.array(result["embedding"]).shape
    assert B == 1 and N == 512

    # Test inference with JSON encoding
    data = {
        "model_id": model_id,
        "method": "encode_text",
        "inputs": {
            "texts": ["A photo of a cat"],
        },
    }
    response = http_client.post(
        "/infer",
        headers={"Content-Type": "application/json"},
        json=encode_dict(data),
    )
    assert response.status_code == 201, response.text
    result = response.json()
    assert isinstance(result, dict)

    assert "embedding" in result
    (B, N) = np.array(result["embedding"]).shape
    assert B == 1 and N == 512


@pytest.mark.parametrize("client_with_server", HTTP_CLIENT_SERVER_CONFIGURATIONS)
def test_http_client_inference_image_generation(client_with_server, request):
    http_client = request.getfixturevalue(client_with_server)
    assert http_client is not None

    from PIL import Image

    from nos.server.http._utils import decode_dict, encode_dict

    # Test inference with JSON encoding
    data = {
        "model_id": "stabilityai/stable-diffusion-2-1",
        "inputs": {
            "prompts": [
                "A photo of a bench on the moon",
            ],
            "num_images": 1,
            "width": 512,
            "height": 512,
        },
    }
    response = http_client.post(
        "/infer",
        headers={"Content-Type": "application/json"},
        json=encode_dict(data),
    )
    assert response.status_code == 201, response.text
    result = response.json()
    assert isinstance(result, dict)
    assert "images" in result

    result = decode_dict(result)
    assert isinstance(result["images"][0], Image.Image)
    assert result["images"][0].size == (512, 512)
