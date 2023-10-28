<center><img src="./docs/assets/nos-header.svg" alt="Nitrous Oxide for your AI Infrastructure"></center>
<p></p>
<p align="center">
<a href="https://pypi.org/project/torch-nos/">
    <img alt="PyPI Version" src="https://badge.fury.io/py/torch-nos.svg">
</a>
<a href="https://pypi.org/project/torch-nos/">
    <img alt="PyPI Version" src="https://img.shields.io/pypi/pyversions/torch-nos">
</a>
<a href="https://www.pepy.tech/projects/torch-nos">
    <img alt="PyPI Downloads" src="https://img.shields.io/pypi/dm/torch-nos">
</a>
<a href="https://github.com/autonomi-ai/nos/blob/main/LICENSE">
    <img alt="PyPi Downloads" src="https://img.shields.io/github/license/torch-nos/torch-nos.svg">
</a><br>
<a href="https://discord.gg/QAGgvTuvgg">
    <img alt="Discord" src="https://img.shields.io/badge/discord-chat-purple?color=%235765F2&label=discord&logo=discord">
</a>
<a href="https://twitter.com/autonomi_ai">
    <img alt="PyPi Version" src="https://img.shields.io/twitter/follow/autonomi_ai.svg?style=social&logo=twitter">
</a>
</p>
<p align="center">
<a href="https://nos.run/"><b>Website</b></a> | <a href="https://docs.nos.run/"><b>Docs</b></a> |  <a href="https://discord.gg/QAGgvTuvgg"><b>Discord</b></a>
</p>

## ⚡️ What is NOS?
**NOS (`torch-nos`)** is a fast and flexible Pytorch inference server, specifically designed for optimizing and running inference of popular foundational AI models.

- 👩‍💻 **Easy-to-use**: Built for [PyTorch](https://pytorch.org/) and designed to optimize, serve and auto-scale Pytorch models in production without compromising on developer experience.
- 🥷 **Flexible**: Run and serve several foundational AI models ([Stable Diffusion](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0), [CLIP](https://huggingface.co/openai/clip-vit-base-patch32), [Whisper](https://huggingface.co/openai/whisper-large-v2)) in a single place.
- 🔌 **Pluggable:** Plug your front-end to NOS with out-of-the-box high-performance gRPC/REST APIs, avoiding all kinds of ML model deployment hassles.
- 🚀 **Scalable**: Optimize and scale models easily for maximum HW performance without a PhD in ML, distributed systems or infrastructure.
- 📦 **Extensible**: Easily hack and add custom models, optimizations, and HW-support in a Python-first environment.
- ⚙️ **HW-accelerated:** Take full advantage of your underlying HW (GPUs, ASICs) without compromise.
- ☁️ **Cloud-agnostic:** Run on any cloud HW (AWS, GCP, Azure, Lambda Labs, On-Prem) with our ready-to-use inference server containers.


> **NOS** inherits its name from **N**itrous **O**xide **S**ystem, the performance-enhancing system typically used in racing cars. NOS is designed to be modular and easy to extend.


## 🚀 Getting Started

Get started with the full NOS server by installing via pip:

  ```shell
  $ conda env create -n nos-py38 python=3.8
  $ conda activate nos-py38
  $ conda install pytorch>=2.0.1 torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
  $ pip install torch-nos[server]
  ```

If you want to simply use a light-weight NOS client and run inference on your local machine (via docker), you can install the client-only package:

  ```shell
  $ conda env create -n nos-py38 python=3.8
  $ conda activate nos-py38
  $ pip install torch-nos
  ```

## 🔥 Quickstart / Show me the code

### Image Generation as-a-Service


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
image, = sdxl(prompts=["fox jumped over the moon"],
              width=1024, height=1024, num_images=1)
```

</td>
<td>


```bash
curl \
-X POST http://localhost:8000/infer \
-H 'Content-Type: application/json' \
-d '{
      "model_id": "stabilityai/stable-diffusion-xl-base-1-0",
      "inputs": {
          "prompts": ["fox jumped over the moon"],
          "width": 1024,
          "height": 1024,
          "num_images": 1
      }
    }'
```

</td>
</tr>
</table>

### Text & Image Embedding-as-a-Service (CLIP-as-a-Service)

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

clip = client.Module("openai/clip")
txt_vec = clip.encode_text(text=["fox jumped over the moon"])
```

</td>
<td>

```bash
curl \
-X POST http://localhost:8000/infer \
-H 'Content-Type: application/json' \
-d '{
      "model_id": "openai/clip",
      "method": "encode_text",
      "inputs": {
          "texts": ["fox jumped over the moon"]
      }
    }'
```

</td>
</tr>
</table>


## 📂 Directory Structure

```bash
├── docker         # Dockerfile for CPU/GPU servers
├── docs           # mkdocs documentation
├── examples       # example guides, jupyter notebooks, demos
├── makefiles      # makefiles for building/testing
├── nos
│   ├── cli        # CLI (hub, system)
│   ├── client     # gRPC / REST client
│   ├── common     # common utilities
│   ├── executors  # runtime executor (i.e. Ray)
│   ├── hub        # hub utilies
│   ├── managers   # model manager / multiplexer
│   ├── models     # model zoo
│   ├── proto      # protobuf defs for NOS gRPC service
│   ├── server     # server backend (gRPC)
│   └── test       # pytest utilities
├── requirements   # requirement extras (server, docs, tests)
├── scripts        # basic scripts
└── tests          # pytests (client, server, benchmark)
```

## 📚 Documentation

- [Quickstart](./docs/quickstart.md)
- [Models](./docs/models/supported-models.md)
- **Concepts**: [NOS Architecture](./docs/concepts/architecture-overview.md)
- **Demos**: [Building a Discord Image Generation Bot](./docs/demos/discord-bot.md), [Video Search Demo](./docs/demos/video-search.md)

## 🛣 Roadmap

### HW / Cloud Support

- [x] **Commodity GPUs**
    - [x] NVIDIA GPUs (20XX, 30XX, 40XX)
    - [ ] AMD GPUs (RX 7000)

- [x] **Cloud GPUs**
    - [x] NVIDIA (H100, A100, A10G, A30G, T4, L4)
    - [ ] AMD (MI200, MI250)

- [x] **Cloud Service Providers** (via [SkyPilot](https://github.com/skypilot-org/skypilot))
    - [x] AWS, GCP, Azure
    - [ ] **Opinionated Cloud:** Lambda Labs, RunPod, etc

- [ ] **Cloud ASICs**
    - [ ] [AWS Inferentia](https://aws.amazon.com/machine-learning/inferentia/) ([Inf1](https://aws.amazon.com/ec2/instance-types/inf1/)/[Inf2](https://aws.amazon.com/ec2/instance-types/inf2/))
    - [ ] Google TPU
    - [ ] Coming soon! (Habana Gaudi, Tenstorrent)


## 📄 License

This project is licensed under the [Apache-2.0 License](LICENSE).

## 📡 Telemetry

NOS collects anonymous usage data using [Sentry](https://sentry.io/). This is used to help us understand how the community is using NOS and to help us prioritize features. You can opt-out of telemetry by setting `NOS_TELEMETRY_ENABLED=0`.

## 🤝 Contributing
We welcome contributions! Please see our [contributing guide](CONTRIBUTING.md) for more information.

### 🔗  Quick Links

* 💬 Send us an email at [support@autonomi.ai](mailto:support@autonomi.ai) or join our [Discord](https://discord.gg/QAGgvTuvgg) for help.
* 📣 Follow us on [Twitter](https://twitter.com/autonomi\_ai), and [LinkedIn](https://www.linkedin.com/company/autonomi-ai) to keep up-to-date on our products.

<style> .md-typeset h1, .md-content__button { display: none; } </style>
