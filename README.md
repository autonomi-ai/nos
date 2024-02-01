<center><img src="./docs/assets/nos-header.svg" alt="Nitrous Oxide for your AI Infrastructure"></center>
<p></p>
<p align="center">
<a href="https://pypi.org/project/torch-nos/"><img alt="PyPI Version" src="https://badge.fury.io/py/torch-nos.svg"></a>
<a href="https://pypi.org/project/torch-nos/"><img alt="PyPI Version" src="https://img.shields.io/pypi/pyversions/torch-nos"></a>
<a href="https://www.pepy.tech/projects/torch-nos"><img alt="PyPI Downloads" src="https://img.shields.io/pypi/dm/torch-nos"></a>
<a href="https://github.com/autonomi-ai/nos/blob/main/LICENSE"><img alt="PyPi Downloads" src="https://img.shields.io/github/license/autonomi-ai/nos.svg"></a><br>
<a href="https://discord.gg/QAGgvTuvgg"><img alt="Discord" src="https://img.shields.io/badge/discord-chat-purple?color=%235765F2&label=discord&logo=discord"></a>
<a href="https://twitter.com/autonomi_ai"><img alt="PyPi Version" src="https://img.shields.io/twitter/follow/autonomi_ai.svg?style=social&logo=twitter"></a>
</p>
<p align="center">
<a href="https://docs.nos.run/"><b>Website</b></a> | <a href="https://docs.nos.run/"><b>Docs</b></a> | <a href="https://docs.nos.run/docs/blog"><b>Blog</b></a> |    <a href="https://discord.gg/QAGgvTuvgg"><b>Discord</b></a>
</p>

## What is NOS?
**NOS (`torch-nos`)** is a fast and flexible Pytorch inference server, specifically designed for optimizing and running inference of popular foundational AI models.
<br>

## 🛠️ **Why use NOS?**

- 👩‍💻 **Easy-to-use**: Built for [PyTorch](https://pytorch.org/) and designed to optimize, serve and auto-scale Pytorch models in production without compromising on developer experience.
- 🥷 **Flexible**: Run and serve several foundational AI models ([Stable Diffusion](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0), [CLIP](https://huggingface.co/openai/clip-vit-base-patch32), [Whisper](https://huggingface.co/openai/whisper-large-v2)) in a single place.
- 🔌 **Pluggable:** Plug your front-end to NOS with out-of-the-box high-performance gRPC/REST APIs, avoiding all kinds of ML model deployment hassles.
- 🚀 **Scalable**: Optimize and scale models easily for maximum HW performance without a PhD in ML, distributed systems or infrastructure.
- 📦 **Extensible**: Easily hack and add custom models, optimizations, and HW-support in a Python-first environment.
- ⚙️ **HW-accelerated:** Take full advantage of your underlying HW (GPUs, ASICs) without compromise.
- ☁️ **Cloud-agnostic:** Run on any cloud HW (AWS, GCP, Azure, Lambda Labs, On-Prem) with our ready-to-use inference server containers.


> **NOS** inherits its name from **N**itrous **O**xide **S**ystem, the performance-enhancing system typically used in racing cars. NOS is designed to be modular and easy to extend.
<br>

## 🔥 What's New

* **[Jan 2024]** ✍️ [blog] [Serving LLMs on a budget](https://docs.nos.run/docs/blog/serving-llms-on-a-budget.html) with [SkyServe](https://skypilot.readthedocs.io/en/latest/serving/sky-serve.html).
* **[Jan 2024]** 📚 [docs] [NOS x SkyPilot Integration](https://docs.nos.run/docs/integrations/skypilot.html) page!
* **[Jan 2024]** ✍️ [blog] [Getting started with NOS tutorials](https://docs.nos.run/docs/blog/-getting-started-with-nos-tutorials.html) is available [here](./examples/tutorials/)!
* **[Dec 2023]** 🛝 [repo] We open-sourced the [NOS playground](https://github.com/autonomi-ai/nos-playground) to help you get started with more examples built on NOS!

## 🚀 Quickstart

We highly recommend that you go to our [quickstart guide](https://docs.nos.run/docs/quickstart.html) to get started. To install the NOS client, you can run the following command:

```bash
conda create -n nos python=3.8
conda activate nos
pip install torch-nos
```

Once the client is installed, you can start the NOS server via the NOS `serve` CLI. This will automatically detect your local environment, download the docker runtime image and spin up the NOS server:

```bash
nos serve up --http
```

You are now ready to run your first inference request with NOS! You can run any of the following commands to try things out.

*Note:* For the above quickstart to work out of the box, we expect the user to have [Docker](https://docs.docker.com/get-docker/), [Nvidia Docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker) and [Docker Compose](https://docs.docker.com/compose/install/) pre-installed on their machine. If you run into any issues, please visit our [quickstart](https://docs.nos.run/docs/quickstart.html) page or ping us on [Discord](https://discord.gg/QAGgvTuvgg).

## 👩‍💻 **What can NOS do?**

### 💬 Chat / LLM Agents (ChatGPT-as-a-Service)
---
NOS provides an OpenAI-compatible server with streaming support so that you can connect your favorite LLM client.

<img src="docs/assets/llama_nos.gif" width="400">

<table>
<tr>
<td> gRPC API ⚡ </td>
<td> REST API </td>
</tr>
<tr>
<td>

```python
from nos.client import Client

client = Client("[::]:50051")

model = client.Module("meta-llama/Llama-2-7b-chat-hf")
response = model.chat(message="Tell me a story of 1000 words with emojis", _stream=True)
```

</td>
<td>

```bash
curl \
-X POST http://localhost:8000/v1/chat/completions \
-H "Content-Type: application/json" \
-d '{
    "model": "meta-llama/Llama-2-7b-chat-hf",
    "messages": [{"role": "user", "content": "Tell me a story of 1000 words with emojis"}],
    "temperature": 0.7, "stream": true
  }'
```

</td>
</tr>
</table>


### 🏞️ Image Generation (Stable-Diffusion-as-a-Service)
---
Build MidJourney discord bots in seconds.

<img src="docs/assets/hippo_with_glasses_sdxl.jpg" width="400">

<table>
<tr>
<td> gRPC API ⚡ </td>
<td> REST API </td>
</tr>
<tr>
<td>

```python
from nos.client import Client

client = Client("[::]:50051")

sdxl = client.Module("stabilityai/stable-diffusion-xl-base-1-0")
image, = sdxl(prompts=["hippo with glasses in a library, cartoon styling"],
              width=1024, height=1024, num_images=1)
```

</td>
<td>


```bash
curl \
-X POST http://localhost:8000/v1/infer \
-H 'Content-Type: application/json' \
-d '{
    "model_id": "stabilityai/stable-diffusion-xl-base-1-0",
    "inputs": {
        "prompts": ["hippo with glasses in a library, cartoon styling"],
        "width": 1024,
        "height": 1024,
        "num_images": 1
    }
}'
```

</td>
</tr>
</table>



### 🧠 Text & Image Embedding (CLIP-as-a-Service)
---
Build scalable semantic search of images/videos in minutes.

<img src="docs/assets/embedding.png" width="400">

<table>
<tr>
<td> gRPC API ⚡ </td>
<td> REST API </td>
</tr>
<tr>
<td>

```python
from nos.client import Client

client = Client("[::]:50051")

clip = client.Module("openai/clip-vit-base-patch32")
txt_vec = clip.encode_text(texts=["fox jumped over the moon"])
```

</td>
<td>

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

</td>
</tr>
</table>



### 🎙️ Audio Transcription (Whisper-as-a-Service)
---

Perform real-time audio transcription using Whisper.

<img src="docs/assets/transcription.png" width="400">

<table>
<tr>
<td> gRPC API ⚡ </td>
<td> REST API </td>
</tr>
<tr>
<td>

```python
from pathlib import Path
from nos.client import Client

client = Client("[::]:50051")

model = client.Module("openai/whisper-small.en")
with client.UploadFile(Path("audio.wav")) as remote_path:
  response = model(path=remote_path)
# {"chunks": ...}
```

</td>
<td>

```bash
curl \
-X POST http://localhost:8000/v1/infer/file \
-H 'accept: application/json' \
-H 'Content-Type: multipart/form-data' \
-F 'model_id=openai/whisper-small.en' \
-F 'file=@audio.wav'
```

</td>
</tr>
</table>

### 🧐 Object Detection (YOLOX-as-a-Service)
---

Run classical computer-vision tasks in 2 lines of code.

<img src="docs/assets/bench_park_detections.png" width="400">

<table>
<tr>
<td> gRPC API ⚡ </td>
<td> REST API </td>
</tr>
<tr>
<td>

```python
from pathlib import Path
from nos.client import Client

client = Client("[::]:50051")

model = client.Module("yolox/medium")
response = model(images=[Image.open("image.jpg")])
```

</td>
<td>

```bash
curl \
-X POST http://localhost:8000/v1/infer/file \
-H 'accept: application/json' \
-H 'Content-Type: multipart/form-data' \
-F 'model_id=yolox/medium' \
-F 'file=@image.jpg'
```

</td>
</tr>
</table>


### ⚒️ Custom models
---
Want to run models not supported by NOS? You can easily add your own models following the examples in the [NOS Playground](https://github.com/autonomi-ai/nos-playground/tree/main/examples).


## 📚 Documentation

- [Tutorials](./examples/tutorials/)
- [Quickstart](https://docs.nos.run/docs/quickstart.html)
- [Models](https://docs.nos.run/docs/models/supported-models.html)
- **Concepts**: [Architecture Overview](https://docs.nos.run/docs/concepts/architecture-overview.html), [ModelSpec](https://docs.nos.run/docs/concepts/model-spec.html), [ModelManager](https://docs.nos.run/docs/concepts/model-manager.html), [Runtime Environments](https://docs.nos.run/docs/concepts/runtime-environments.html)
- **Demos**: [Building a Discord Image Generation Bot](https://docs.nos.run/docs/demos/discord-bot.html), [Video Search Demo](https://docs.nos.run/docs/demos/video-search.html)

## 📄 License

This project is licensed under the [Apache-2.0 License](LICENSE).

## 📡 Telemetry

NOS collects anonymous usage data using [Sentry](https://sentry.io/). This is used to help us understand how the community is using NOS and to help us prioritize features. You can opt-out of telemetry by setting `NOS_TELEMETRY_ENABLED=0`.

## 🤝 Contributing
We welcome contributions! Please see our [contributing guide](CONTRIBUTING.md) for more information.

### 🔗  Quick Links

* 💬 Send us an email at [support@autonomi.ai](mailto:support@autonomi.ai) or join our [Discord](https://discord.gg/QAGgvTuvgg) for help.
* 📣 Follow us on [Twitter](https://twitter.com/autonomi\_ai), and [LinkedIn](https://www.linkedin.com/company/autonomi-ai) to keep up-to-date on our products.

<br>
<style> .md-typeset h1, .md-content__button { display: none; } </style>
