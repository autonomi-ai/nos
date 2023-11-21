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
    response = http_client.get("/v1/health")
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
    model_id = "yolox/small"
    data = {
        "model_id": model_id,
        "inputs": {
            "images": Image.open(NOS_TEST_IMAGE),
        },
    }
    response = http_client.post(
        "/v1/infer",
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

    # Test inference with file upload
    response = http_client.post(
        "/v1/infer/file",
        headers={"accept": "application/json"},
        files={
            "model_id": (None, model_id),
            "file": NOS_TEST_IMAGE.open("rb"),
        },
    )
    assert response.status_code == 201, response.text
    result = response.json()
    assert isinstance(result, dict)

    # Test inference with URL
    response = http_client.post(
        "/v1/infer/file",
        headers={"accept": "application/json"},
        files={
            "model_id": (None, model_id),
            "url": (None, "https://raw.githubusercontent.com/autonomi-ai/nos/main/nos/test/test_data/test.jpg"),
        },
    )
    assert response.status_code == 201, response.text
    result = response.json()
    assert isinstance(result, dict)


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
        "/v1/infer",
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
        "/v1/infer",
        headers={"Content-Type": "application/json"},
        json=encode_dict(data),
    )
    assert response.status_code == 201, response.text
    result = response.json()
    assert isinstance(result, dict)

    assert "embedding" in result
    (B, N) = np.array(result["embedding"]).shape
    assert B == 1 and N == 512

    # Test inference with file upload
    response = http_client.post(
        "/v1/infer/file",
        headers={"accept": "application/json"},
        files={
            "model_id": (None, model_id),
            "method": (None, "encode_image"),
            "file": NOS_TEST_IMAGE.open("rb"),
        },
    )
    assert response.status_code == 201, response.text
    result = response.json()
    assert isinstance(result, dict)

    # Test inference with URL
    response = http_client.post(
        "/v1/infer/file",
        headers={"accept": "application/json"},
        files={
            "model_id": (None, model_id),
            "method": (None, "encode_image"),
            "url": (None, "https://raw.githubusercontent.com/autonomi-ai/nos/main/nos/test/test_data/test.jpg"),
        },
    )
    assert response.status_code == 201, response.text
    result = response.json()
    assert isinstance(result, dict)


@pytest.mark.parametrize("client_with_server", HTTP_CLIENT_SERVER_CONFIGURATIONS)
def test_http_client_inference_image_generation(client_with_server, request):
    http_client = request.getfixturevalue(client_with_server)
    assert http_client is not None

    from PIL import Image

    from nos.server.http._utils import decode_dict, encode_dict

    # Test inference with JSON encoding
    model_id = "stabilityai/stable-diffusion-2-1"
    data = {
        "model_id": model_id,
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
        "/v1/infer",
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


@pytest.mark.parametrize("client_with_server", HTTP_CLIENT_SERVER_CONFIGURATIONS)
def test_http_client_whisper(client_with_server, request):
    http_client = request.getfixturevalue(client_with_server)
    assert http_client is not None

    from nos.test.utils import NOS_TEST_AUDIO

    # Test inference with file upload
    model_id = "openai/whisper-tiny.en"
    response = http_client.post(
        "/v1/infer/file",
        headers={"accept": "application/json"},
        files={
            "model_id": (None, model_id),
            "file": NOS_TEST_AUDIO.open("rb"),
        },
    )
    assert response.status_code == 201, response.text
    result = response.json()
    assert isinstance(result, dict)
    assert "chunks" in result

    # Test inference with URL
    model_id = "openai/whisper-tiny.en"
    response = http_client.post(
        "/v1/infer/file",
        headers={"accept": "application/json"},
        files={
            "model_id": (None, model_id),
            "url": (
                None,
                "https://raw.githubusercontent.com/autonomi-ai/nos/main/nos/test/test_data/test_speech.flac",
            ),
        },
    )
    assert response.status_code == 201, response.text
    result = response.json()
    assert isinstance(result, dict)
    assert "chunks" in result


@pytest.mark.parametrize("client_with_server", HTTP_CLIENT_SERVER_CONFIGURATIONS)
def test_http_client_chat(client_with_server, request):
    http_client = request.getfixturevalue(client_with_server)
    assert http_client is not None

    response = http_client.get(
        "/v1/chat/models",
        headers={"accept": "application/json"},
    )
    assert response.status_code == 200, response.text
    assert isinstance(response.json(), list)
    for model in response.json():
        assert "id" in model
        assert "object" in model

    model_id = "HuggingFaceH4/tiny-random-LlamaForCausalLM"
    with http_client.stream(
        "POST",
        "/v1/chat/completions",
        headers={"accept": "application/json"},
        json={
            "model": model_id,
            "messages": [
                {"role": "user", "content": "What is the meaning of life?"},
                {"role": "assistant", "content": "I'm not sure I understand"},
                {
                    "role": "user",
                    "content": "You're a sage who has spent 10 thousand hours on the meaning of life. What is the meaning of life?",
                },
            ],
            "max_tokens": 512,
            "temperature": 0.7,
        },
    ) as response:
        # Parse the text/event-stream response
        for chunk in response.iter_raw():
            print(chunk.decode("utf-8"))
