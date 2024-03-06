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
pytest -sv ./tests/test_sdxl_client.py
```

### Call the service

You can also call the service via the REST API directly:

```bash
curl \
-X POST http://<service-ip>:8000/v1/infer \
-H 'Content-Type: application/json' \
-d '{
    "model_id": "stabilityai/stable-diffusion-xl-base-1.0-inf2",
    "inputs": {
        "prompts": ["hippo with glasses in a library, cartoon styling"],
        "width": 1024, "height": 1024,
        "num_images": 1,
        "num_inference_steps": 50,
        "guidance_scale": 7.5,
        "num_images": 1
    }
}'
```
