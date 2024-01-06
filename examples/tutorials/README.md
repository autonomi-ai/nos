## 📦 NOS Tutorials

The following tutorials give a brief overview of how to use NOS to serve models.

- [x] [`01-serving-custom-models`](./01-serving-custom-models): Serve a custom GPU model with NOS.
- [x] [`02-serving-custom-methods`](./02-serving-custom-methods): Expose several custom methods of a model for serving purposes.
- [x] [`03-llm-streaming-chat`](./03-llm-streaming-chat): Serve an LLM with *streaming* support ([`TinyLlama/TinyLlama-1.1B-Chat-v0.1`](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v0.1)).


## 🏃‍♂️ Running the examples

For each of the examples, you can run the following command to serve the model (in one of your terminals):

```bash
nos serve up -c serve.yaml
```

You can then run the tests in the `tests` directory to check if the model is served correctly:

```bash
pytest -sv ./tests
```

For HTTP tests, you'll need add the `--http` flag to the `nos serve` command:

```bash
nos serve up -c serve.yaml --http
```
