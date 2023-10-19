<h1 align="center" style="font-size:64px;font-weight: bold;font-color:black;">⚡️ NOS</h1>
<h4 align="center">Nitrous Oxide System for your AI Infrastructure
<p style="font-weight: normal;">
Optimize, serve and auto-scale Pytorch models in production<br>
</p>
</h4>


<p align="center">
<a href="https://nos.run/"><b>Website</b></a> | <a href="https://docs.nos.run/"><b>Docs</b></a> |  <a href="https://discord.gg/QAGgvTuvgg"><b>Discord</b></a>
</p>

<p align="center">
<a href="https://pypi.org/project/torch-nos/">
    <img alt="PyPi Version" src="https://badge.fury.io/py/torch-nos.svg">
</a>
<a href="https://pypi.org/project/torch-nos/">
    <img alt="PyPi Version" src="https://img.shields.io/pypi/pyversions/torch-nos">
</a>
<a href="https://pypi.org/project/torch-nos/">
    <img alt="PyPi Downloads" src="https://img.shields.io/pypi/dm/torch-nos">
</a>
<a href="https://discord.gg/QAGgvTuvgg">
    <img alt="Discord" src="https://img.shields.io/badge/discord-chat-purple?color=%235765F2&label=discord&logo=discord">
</a>
<a href="https://twitter.com/autonomi_ai">
    <img alt="PyPi Version" src="https://img.shields.io/twitter/follow/autonomi_ai.svg?style=social&logo=twitter">
</a>

</p>


**NOS (`torch-nos`)** is a fast and flexible Pytorch inference server, specifically designed for optimizing and running lightning-fast inference of popular foundational AI models.

*Optimizing and serving models for production AI inference is still difficult, often leading to notoriously expensive cloud bills and often underutilized GPUs. That’s why we’re building **NOS** - a fast inference server for modern AI workloads. With a few lines of code, developers can optimize, serve, and auto-scale Pytorch model inference without having to deal with the complexities of ML compilers, HW-accelerators, or distributed inference. Simply put, NOS allows AI teams to cut inference costs up to **10x**, speeding up development time and time-to-market.*


## 👩‍💻 What is NOS?
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
conda env create -n nos-py38 python=3.8
conda activate nos-py38
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install torch-nos[server]
```

## 📂 Repository Structure

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

- 📚 [NOS Documentation](https://docs.nos.run/)
- 🔥 [Quickstart](https://docs.nos.run/docs/quickstart.html)
- 🧠 [Models](https://docs.nos.run/docs/models/supported-models.html)
- ⚡️ **Concepts**: [NOS Architecture](https://docs.nos.run/docs/concepts/architecture-overview.html)
- 🤖 **Demos**: [Building a Discord Image Generation Bot](https://docs.nos.run/docs/demos/discord-bot.html), [Video Search Demo](https://docs.nos.run/docs/demos/video-search.html)

## 🛣 Roadmap

### HW / Cloud Support

- [✅] **Commodity GPUs**
  - [✅] NVIDIA GPUs (20XX, 30XX, 40XX)
  - [ ] AMD GPUs (RX 6000 series)
- [✅] **Cloud GPUs**
  - [ ] NVIDIA (T4, A100, H100)
  - [-] AMD (MI200, MI250)
- [🟡] **Cloud ASICs**
  - [🟡] [AWS Inferentia](https://aws.amazon.com/machine-learning/inferentia/) [Inf1](https://aws.amazon.com/ec2/instance-types/inf1/)/[Inf2](https://aws.amazon.com/ec2/instance-types/inf2/)
  - [ ] Google TPU
  - [ ] TBD (Graphcore, Habana Gaudi, Tenstorrent)
- [✅] **Cloud Service Providers** (via [SkyPilot](https://github.com/skypilot-org/skypilot))
    - [✅] **Big 3:** AWS, GCP, Azure
    - [ ] **Opinionated Cloud:** Lambda Labs, RunPod, etc


## 📄 License

This project is licensed under the [Apache-2.0 License](LICENSE).


## 🤝 Contributing
We welcome contributions! Please see our [contributing guide](CONTRIBUTING.md) for more information.

### 🔗  Quick Links

* 💬 Send us an email at [support@autonomi.ai](mailto:support@autonomi.ai) or join our [Discord](https://discord.gg/QAGgvTuvgg) for help.
* 📣 Follow us on [Twitter](https://twitter.com/autonomi\_ai), and [LinkedIn](https://www.linkedin.com/company/autonomi-ai) to keep up-to-date on our products.