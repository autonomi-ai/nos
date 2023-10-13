import pytest

from nos.test.utils import NOS_TEST_IMAGE


pytestmark = pytest.mark.skip


# TODO (spillai): Add support for "local", "cpu", "gpu" and "auto" runtimes
# @pytest.mark.parametrize("runtime", ["cpu", "gpu", "auto"])
def test_http_client(local_http_client_with_server):  # noqa: F811
    http_client = local_http_client_with_server
    assert http_client is not None

    # Health check
    response = http_client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_http_client_inference_object_detection_2d(local_http_client_with_server):
    http_client = local_http_client_with_server
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


def test_http_client_inference_image_embedding(local_http_client_with_server):
    http_client = local_http_client_with_server
    assert http_client is not None

    import numpy as np
    from PIL import Image

    from nos.server.http._utils import encode_dict

    # Test inference with JSON encoding
    data = {
        "model_id": "openai/clip",
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


def test_http_client_inference_image_generation(local_http_client_with_server):
    http_client = local_http_client_with_server
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
