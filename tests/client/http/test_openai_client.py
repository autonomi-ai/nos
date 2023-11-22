import pytest

from nos.logging import logger
from nos.test.conftest import (  # noqa: F401, E402
    HTTP_TEST_PORT,
    http_server_with_gpu_backend,
)


@pytest.mark.skipif(pytest.importorskip("openai") is None, reason="openai is not installed")
@pytest.mark.client
def test_openai_client_chat_completion(http_server_with_gpu_backend):  # noqa: F811
    import time

    import openai
    import requests
    from tqdm import tqdm

    # Check health
    BASE_URL = f"http://localhost:{HTTP_TEST_PORT}/v1"
    response = requests.get(f"{BASE_URL}/health")
    assert response.status_code == 200

    model = "meta-llama--Llama-2-7b-chat-hf"
    client = openai.OpenAI(base_url=BASE_URL, api_key="sk-fake-key")
    assert client is not None

    # List all models
    models = client.models.list()
    assert models is not None
    print(models)

    # Create a chat completion prompt
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell a story in less than 300 words."},
    ]

    # Test chat completion (non-streaming)
    st = time.time()
    chat_completion = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.7,
    )
    logger.debug(f"chat completion took {(time.time() - st):.3f} seconds to complete")
    assert chat_completion is not None
    assert isinstance(chat_completion, openai.types.chat.chat_completion.ChatCompletion)

    # Test chat completion (streaming)
    st = time.time()
    chat_completion = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.7,
        stream=True,
    )
    for chunk in tqdm(chat_completion, desc="Streaming chat completion"):
        if st is not None:
            logger.debug(f"chat completion took {time.time() - st:.3f} seconds to start streaming")
            st = None
        assert chunk is not None
        assert isinstance(chunk, openai.types.chat.chat_completion_chunk.ChatCompletionChunk)
