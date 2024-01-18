## Serving Custom Models with multiple methods

This tutorial shows how to serve multiple methods of a custom model with NOS.

### Serve the model

The `serve.yaml` file contains the specification of the custom image that will be used to build the docker runtime image and serve the model using this custom rumtime image. You can serve the model via:

```bash
nos serve up -c serve.yaml
```

**Note:** *You will notice that in this example, we are deploying the model on the CPU. If you want to deploy the model on GPUs, you can change the `deployment.resources.device` to `gpu` in the `serve.yaml`.*


### Run the tests (via the gRPC client)

You can now run the tests to check that the model is served correctly:

```bash
python tests/test_model.py
```

### Call the model (via the REST API)

You can also call the model via the REST API. In order to do so, you need to start the server with `--http` flag to enable the HTTP proxy.

```bash
nos serve up -c serve.yaml --http
```

You can then call the model's specific method via:

```bash
curl \
-X POST http://localhost:8000/v1/infer \
-H 'Content-Type: application/json' \
-d '{
    "model_id": "custom/clip-model-cpu",
    "method": "encode_text",
    "inputs": {
        "texts": ["fox jumped over the moon"]
    }
}'
```
