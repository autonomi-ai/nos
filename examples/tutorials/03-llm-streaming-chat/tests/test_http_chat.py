import sys


HTTP_PORT = 8000
model_id = "tinyllama-1.1b-chat"


if __name__ == "__main__":
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
