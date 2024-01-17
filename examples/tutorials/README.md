## 📦 NOS Tutorials

The following tutorials give a brief overview of how to use NOS to serve models.

- [x] [`01-serving-custom-models`](./01-serving-custom-models): Serve a custom GPU model with NOS.
- [x] [`02-serving-multiple-methods`](./02-serving-multiple-methods): Expose several custom methods of a model for serving purposes.
- [x] [`03-llm-streaming-chat`](./03-llm-streaming-chat): Serve an LLM with *streaming* support ([`TinyLlama/TinyLlama-1.1B-Chat-v0.1`](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v0.1)).
- [x] [`04-serving-multiple-models`](./04-serving-multiple-models): Serve multiple models such as [`TinyLlama/TinyLlama-1.1B-Chat-v0.1`](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0) and [distil-whisper/distil-small.en](https://huggingface.co/distil-whisper/distil-small.en) on the same GPU with custom model resources -- enable multi-modal applications like [audio transcription + summarization](./04-serving-multiple-models/summarize_audio.py) on the same device.
- [x] [`05-serving-with-docker`](./05-serving-with-docker): Use NOS in a production environment with Docker and Docker Compose.

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
