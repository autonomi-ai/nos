## Serving Custom Models

This tutorial shows how to serve a custom model with NOS.

### Serve the model

The `serve.yaml` file contains the specification of the custom image that will be used to build the docker runtime image and serve the model using this custom rumtime image. You can serve the model via:

```bash
nos serve up -c serve.yaml
```

### Run the tests (via the gRPC client)

You can now run the tests to check that the model is served correctly:

```bash
python tests/test_model.py
```
