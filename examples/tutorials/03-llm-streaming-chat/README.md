## Serving LLMs with streaming support

This tutorial shows how to serve an LLM with streaming support.

### Serve the model

The `serve.yaml` file contains the specification of the custom image that will be used to build the docker runtime image and serve the model using this custom rumtime image. You can serve the model via:

```bash
nos serve up -c serve.yaml --http
```

### Run the tests (via the gRPC client)

You can now run the tests to check that the model is served correctly:

```bash
python tests/test_grpc_chat.py
```

### Run the tests (via the REST/HTTP client)

You can also run the tests to check that the model is served correctly via the REST API:

```bash
python tests/test_http_chat.py
```

## Use cURL to call the model (via the REST API)

NOS also exposes an OpenAI API compatible endpoint for such custom LLM models. You can call the model via the `/chat/completions` route:

```bash
curl \
-X POST http://localhost:8000/v1/chat/completions \
-H "Content-Type: application/json" \
-d '{
    "model": "tinyllama-1.1b-chat",
    "messages": [{"role": "user", "content": "Tell me a story of 1000 words with emojis"}],
    "temperature": 0.7, "stream": true
  }'
```
