<center><img src="./docs/assets/nos-header.svg" alt="Nitro Boost for your AI Infrastructure"></center>
<p></p>
<p align="center">
<a href="https://docs.nos.run/"><b>Website</b></a> | <a href="https://docs.nos.run/"><b>Docs</b></a> | <a href="https://github.com/autonomi-ai/nos/tree/main/examples/tutorials"><b>Tutorials</b></a> | <a href="https://github.com/autonomi-ai/nos-playground"><b>Playground</b></a> | <a href="https://docs.nos.run/docs/blog"><b>Blog</b></a> | <a href="https://discord.gg/QAGgvTuvgg"><b>Discord</b></a>
</p>
<p align="center">
<a href="https://pypi.org/project/torch-nos/"><img alt="PyPI Version" src="https://badge.fury.io/py/torch-nos.svg"></a>
<a href="https://pypi.org/project/torch-nos/"><img alt="PyPI Version" src="https://img.shields.io/pypi/pyversions/torch-nos"></a>
<a href="https://www.pepy.tech/projects/torch-nos"><img alt="PyPI Downloads" src="https://img.shields.io/pypi/dm/torch-nos"></a>
<a href="https://hub.docker.com/repository/docker/autonomi/nos/general"><img alt="Docker Pulls" src="https://img.shields.io/docker/pulls/autonomi/nos.svg"></a><br>
<a href="https://github.com/autonomi-ai/nos/blob/main/LICENSE"><img alt="PyPi Downloads" src="https://img.shields.io/github/license/autonomi-ai/nos.svg"></a>
<a href="https://discord.gg/QAGgvTuvgg"><img alt="Discord" src="https://img.shields.io/badge/discord-chat-purple?color=%235765F2&label=discord&logo=discord"></a>
<a href="https://twitter.com/autonomi_ai"><img alt="PyPi Version" src="https://img.shields.io/twitter/follow/autonomi_ai.svg?style=social&logo=twitter"></a>
</p>

**NOS** is a fast and flexible PyTorch inference server that runs on any cloud or AI HW.

## 🛠️ Key Features

- 👩‍💻 **Easy-to-use**: Built for [PyTorch](https://pytorch.org/) and designed to optimize, serve and auto-scale Pytorch models in production without compromising on developer experience.
- 🥷 **Multi-modal & Multi-model**: Serve multiple foundational AI models ([LLMs](https://github.com/autonomi-ai/nos/blob/main/nos/models/llm.py), [Diffusion](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0), [Embeddings](https://github.com/autonomi-ai/nos/blob/main/nos/models/clip.py), [Speech-to-Text](https://github.com/autonomi-ai/nos/blob/main/nos/models/clip.py) and [Object Detection](https://github.com/autonomi-ai/nos/blob/main/nos/models/yolox.py)) simultaneously, in a single server.
- ⚙️ **HW-aware Runtime:** Deploy PyTorch models effortlessly on modern AI accelerators (NVIDIA GPUs, AWS Inferentia2, AMD - coming soon, and even CPUs).
- ☁️ **Cloud-agnostic Containers:** Run on any cloud (AWS, GCP, Azure, Lambda Labs, On-Prem) with our ready-to-use inference server containers.

## 🔥 What's New

* **[Feb 2024]** ✍️ [blog] [Introducing the NOS Inferentia2 (`inf2`) runtime](https://docs.nos.run/docs/blog/introducing-the-nos-inferentia2-runtime.html).
* **[Jan 2024]** ✍️ [blog] [Serving LLMs on a budget](https://docs.nos.run/docs/blog/serving-llms-on-a-budget.html) with [SkyServe](https://skypilot.readthedocs.io/en/latest/serving/sky-serve.html).
* **[Jan 2024]** 📚 [docs] [NOS x SkyPilot Integration](https://docs.nos.run/docs/integrations/skypilot.html) page!
* **[Jan 2024]** ✍️ [blog] [Getting started with NOS tutorials](https://docs.nos.run/docs/blog/-getting-started-with-nos-tutorials.html) is available [here](./examples/tutorials/)!
* **[Dec 2023]** 🛝 [repo] We open-sourced the [NOS playground](https://github.com/autonomi-ai/nos-playground) to help you get started with more examples built on NOS!

## 🚀 Quickstart

We highly recommend that you go to our [quickstart guide](https://docs.nos.run/docs/quickstart.html) to get started. To install the NOS client, you can run the following command:

```bash
conda create -n nos python=3.8 -y
conda activate nos
pip install torch-nos
```

Once the client is installed, you can start the NOS server via the NOS `serve` CLI. This will automatically detect your local environment, download the docker runtime image and spin up the NOS server:

```bash
nos serve up --http --logging-level INFO
```

You are now ready to run your [first inference request](#👩‍💻-what-can-nos-do) with NOS! You can run any of the following commands to try things out. You can set the logging level to `DEBUG` if you want more detailed information from the server.

## 👩‍💻 **What can NOS do?**

### 💬 Chat / LLM Agents (ChatGPT-as-a-Service)
---
NOS provides an OpenAI-compatible server with streaming support so that you can connect your favorite OpenAI-compatible LLM client to talk to NOS.

<img src="docs/assets/llama_nos.gif" width="400">

<br>
<details>
<summary> API / Usage</summary>
<br>

<b>gRPC API ⚡</b>
```python
from nos.client import Client

client = Client("[::]:50051")

model = client.Module("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
response = model.chat(message="Tell me a story of 1000 words with emojis", _stream=True)
```

<b>REST API</b>
```bash
curl \
-X POST http://localhost:8000/v1/chat/completions \
-H "Content-Type: application/json" \
-d '{
    "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "messages": [{
        "role": "user",
        "content": "Tell me a story of 1000 words with emojis"
    }],
    "temperature": 0.7,
    "stream": true
  }'
```

</details>

### 🏞️ Image Generation (Stable-Diffusion-as-a-Service)
---
Build MidJourney discord bots in seconds.

<img src="docs/assets/hippo_with_glasses_sdxl.jpg" width="400">

<br>
<details>
<summary> API / Usage</summary>
<br>

<b>gRPC API ⚡</b>

```python
from nos.client import Client

client = Client("[::]:50051")

sdxl = client.Module("stabilityai/stable-diffusion-xl-base-1-0")
image, = sdxl(prompts=["hippo with glasses in a library, cartoon styling"],
              width=1024, height=1024, num_images=1)
```

<b>REST API</b>

```bash
curl \
-X POST http://localhost:8000/v1/infer \
-H 'Content-Type: application/json' \
-d '{
    "model_id": "stabilityai/stable-diffusion-xl-base-1-0",
    "inputs": {
        "prompts": ["hippo with glasses in a library, cartoon styling"],
        "width": 1024, "height": 1024,
        "num_images": 1
    }
}'
```

</details>

### 🧠 Text & Image Embedding (CLIP-as-a-Service)
---
Build [scalable semantic search of images/videos](https://docs.nos.run/docs/demos/video-search.html) in minutes.

<img src="docs/assets/embedding.png" width="400">

<br>
<details>
<summary> API / Usage</summary>
<br>

<b>gRPC API ⚡</b>

```python
from nos.client import Client

client = Client("[::]:50051")

clip = client.Module("openai/clip-vit-base-patch32")
txt_vec = clip.encode_text(texts=["fox jumped over the moon"])
```

<b>REST API</b>

```bash
curl \
-X POST http://localhost:8000/v1/infer \
-H 'Content-Type: application/json' \
-d '{
    "model_id": "openai/clip-vit-base-patch32",
    "method": "encode_text",
    "inputs": {
        "texts": ["fox jumped over the moon"]
    }
}'
```

</details>


### 🎙️ Audio Transcription (Whisper-as-a-Service)
---
Perform [real-time audio transcription](./examples/tutorials/04-serving-multiple-models/) using Whisper.

<img src="docs/assets/transcription.png" width="400">

<br>
<details>
<summary> API / Usage</summary>
<br>

<b>gRPC API ⚡</b>

```python
from pathlib import Path
from nos.client import Client

client = Client("[::]:50051")

model = client.Module("openai/whisper-small.en")
with client.UploadFile(Path("audio.wav")) as remote_path:
  response = model(path=remote_path)
# {"chunks": ...}
```

<b>REST API</b>

```bash
curl \
-X POST http://localhost:8000/v1/infer/file \
-H 'accept: application/json' \
-H 'Content-Type: multipart/form-data' \
-F 'model_id=openai/whisper-small.en' \
-F 'file=@audio.wav'
```

</details>

### 🧐 Object Detection (YOLOX-as-a-Service)
---
Run classical computer-vision tasks in 2 lines of code.

<img src="docs/assets/bench_park_detections.png" width="400">

<br>
<details>
<summary> API / Usage</summary>
<br>

<b>gRPC API ⚡</b>

```python
from pathlib import Path
from nos.client import Client

client = Client("[::]:50051")

model = client.Module("yolox/medium")
response = model(images=[Image.open("image.jpg")])
```

<b>REST API</b>

```bash
curl \
-X POST http://localhost:8000/v1/infer/file \
-H 'accept: application/json' \
-H 'Content-Type: multipart/form-data' \
-F 'model_id=yolox/medium' \
-F 'file=@image.jpg'
```

</details>

### ⚒️ Custom models
---
Want to run models not supported by NOS? You can easily add your own models following the examples in the [NOS Playground](https://github.com/autonomi-ai/nos-playground/tree/main/examples).

## 📄 License

This project is licensed under the [Apache-2.0 License](LICENSE).

## 📡 Telemetry

NOS collects anonymous usage data using [Sentry](https://sentry.io/). This is used to help us understand how the community is using NOS and to help us prioritize features. You can opt-out of telemetry by setting `NOS_TELEMETRY_ENABLED=0`.

## 🤝 Contributing
We welcome contributions! Please see our [contributing guide](CONTRIBUTING.md) for more information.

## 🔗  Quick Links

* 💬 Send us an email at [support@autonomi.ai](mailto:support@autonomi.ai) or join our [Discord](https://discord.gg/QAGgvTuvgg) for help.
* 📣 Follow us on [Twitter](https://twitter.com/autonomi\_ai), and [LinkedIn](https://www.linkedin.com/company/autonomi-ai) to keep up-to-date on our products.

<br>
<style> .md-typeset h1, .md-content__button { display: none; } </style>
