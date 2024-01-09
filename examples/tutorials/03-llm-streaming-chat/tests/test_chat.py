import sys

import pytest
from rich import print

from nos.client import Client


HTTP_PORT = 8000
GRPC_PORT = 50051
MODELS = [
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
]


@pytest.fixture(scope="session", autouse=True)
def client():
    client = Client(f"[::]:{GRPC_PORT}")
    assert client.WaitForServer()
    yield client


@pytest.mark.parametrize("model_id", MODELS)
def test_streaming_chat_grpc(client, model_id):

    # Load the llama chat model
    model = client.Module(model_id)

    # Chat with the model
    query = "What is the meaning of life?"

    print()
    print("-" * 80)
    print(f">>> Chatting with the model (model={model_id}) ...")
    print(f"[bold yellow]Query: {query}[/bold yellow]")
    messages = [
        {"role": "system", "content": ""},
        {"role": "user", "content": query},
    ]
    for response in model.chat(messages=messages, max_new_tokens=1024, _stream=True):
        sys.stdout.write(response)
        sys.stdout.flush()
    print()


@pytest.mark.parametrize("model_id", MODELS)
def test_streaming_chat_http(model_id):
    import requests

    BASE_URL = f"http://localhost:{HTTP_PORT}"

    # model_id for LLMs are normalized replacing "/" with "--"
    model_id = model_id.replace("/", "--")

    http_client = requests.Session()
    response = http_client.get(
        f"{BASE_URL}/v1/models",
        headers={"accept": "application/json"},
    )
    assert response.status_code == 200
    assert isinstance(response.json(), dict)
    models = response.json()["data"]
    for model in models:
        assert "id" in model
        assert "object" in model
        response = http_client.get(
            f"{BASE_URL}/v1/models/{model['id']}",
            headers={"accept": "application/json"},
        )
        assert response.status_code == 200, "Failed to get model info: {}".format(model["id"])

    with http_client.post(
        f"{BASE_URL}/v1/chat/completions",
        headers={"accept": "application/json"},
        json={
            "model": model_id,
            "messages": [
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": "What is the meaning of life?"},
            ],
            "max_tokens": 512,
            "temperature": 0.7,
            "stream": True,
        },
        stream=True,
    ) as response:
        for chunk in response.iter_content():
            sys.stdout.write(chunk.decode("utf-8"))
            sys.stdout.flush()


@pytest.mark.skip(reason="Not implemented")
@pytest.mark.parametrize("model_id", MODELS)
def test_streaming_chat_openai(model_id):
    import time

    import openai
    import requests

    # model_id for LLMs are normalized replacing "/" with "--"
    model_id = model_id.replace("/", "--")

    # Check health
    BASE_URL = f"http://localhost:{HTTP_PORT}/v1"
    response = requests.get(f"{BASE_URL}/health")
    assert response.status_code == 200

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
        {"role": "user", "content": "Make it funny."},
    ]

    # Test chat completion (non-streaming)
    st = time.time()
    chat_completion = client.chat.completions.create(
        model=model_id,
        messages=messages,
        temperature=0.7,
    )
    print(f"chat completion took {(time.time() - st):.3f} seconds to complete")
    assert chat_completion is not None
    assert isinstance(chat_completion, openai.types.chat.chat_completion.ChatCompletion)

    # Test chat completion (streaming)
    st = time.time()
    for chunk in client.chat.completions.create(
        model=model_id,
        messages=messages,
        temperature=0.7,
        stream=True,
    ):
        if st is not None:
            print(f"chat completion took {time.time() - st:.3f} seconds to start streaming")
            st = None
        assert chunk is not None
        assert isinstance(chunk, openai.types.chat.chat_completion_chunk.ChatCompletionChunk)
        assert chunk.choices is not None
        assert isinstance(chunk.choices, list)
        assert isinstance(chunk.choices[0].delta, openai.types.chat.chat_completion_chunk.ChoiceDelta)
