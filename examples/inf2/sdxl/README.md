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
pytest -sv ./tests/test_sdxl_inf2_client.py
```

### Call the service

You can also call the service via the REST API directly:

```bash
curl \
-X POST http://127.0.0.1:8000/v1/infer \
-H 'Content-Type: application/json' \
-d '{
    "model_id": "stabilityai/stable-diffusion-xl-base-1.0-inf2",
    "inputs": {
        "prompts": ["fox jumped over the moon"],
        "height": 512,
        "width": 512,
        "num_images": 1,
        "num_inference_steps": 50,
        "guidance_scale": 7.5
    }
}'
```
