## Embeddings Service

Start the server via:
```bash
nos serve up -c serve.yaml --http
```

Optionally, you can provide the `inf2` runtime flag, but this is automatically inferred.

```bash
nos serve up -c serve.yaml --http --runtime inf2
```

### Run the tests

```bash
pytest -sv ./tests/test_embeddings_client.py
```

### Call the service

You can also call the service via the REST API directly:

```bash
curl \
-X POST http://<service-ip>:8000/v1/infer \
-H 'Content-Type: application/json' \
-d '{
    "model_id": "BAAI/bge-small-en-v1.5",
    "inputs": {
        "texts": ["fox jumped over the moon"]
    }
}'
```
