import pytest

from nos.logging import logger
from nos.test.conftest import (  # noqa: F401, E402
    HTTP_TEST_PORT,
    http_server_with_gpu_backend,
)


def openai_v1_available():
    try:
        import openai

        version = openai.__version__.split(".")
        return version[0] >= "1"
    except ImportError:
        return False


pytestmark = pytest.mark.skipif(not openai_v1_available(), reason="openai version >= 1.x.x")


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
    # Note (spllai): Some APIs provide multiple user messages, so
    # we'll need to test this to make sure that the chat template
    # doesn't raise an error.
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell a story in less than 300 words."},
        {"role": "user", "content": "Make it funny."},
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
        assert chunk.choices is not None
        assert isinstance(chunk.choices, list)
        assert isinstance(chunk.choices[0].delta, openai.types.chat.chat_completion_chunk.ChoiceDelta)
