<h1 align="center" style="font-size:54px;font-weight: bold;font-color:black;">🔥 <i>NOS</i></h1>
<h4 align="center">
Nitrous Oxide System for AI. <br> 
Optimize, serve and auto-scale Pytorch models on any hardware. <br>
Cut your inference costs by 10x.
</h4>


<p align="center">
<a href="https://www.autonomi.ai/"><b>Website</b></a> | <a href="https://autonomi-ai.github.io/nos/"><b>Docs</b></a> |  <a href="https://discord.gg/QAGgvTuvgg"><b>Discord</b></a>
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


**NOS** is a PyTorch library for optimizing and running lightning-fast inference of popular computer vision models.

*Optimizing and serving models for production AI inference is still difficult, often leading to notoriously expensive cloud bills and often underutilized GPUs. That’s why we’re building **NOS** - a fast inference server for modern AI workloads. With a few lines of code, developers can optimize, serve, and auto-scale Pytorch model inference without having to deal with the complexities of ML compilers, HW-accelerators, or distributed inference. Simply put, NOS allows AI teams to cut inference costs up to **10x**, speeding up development time and time-to-market.*


## What is NOS?
- ⚡️ **Fast**: Built for PyTorch and designed to optimize/run models faster
- 🔥 **Performant**: Run models such as SDv2 or object detection 2-3x faster out-of-the-box
- 👩‍💻 **No PhD required**: Optimize models for maximum HW performance without a PhD in ML
- 📦 **Extensible**: Easily add optimization and HW-support for custom models
- ⚙️ **HW-accelerated:** Take full advantage of your HW (GPUs, ASICs) without compromise
- ☁️ **Cloud-agnostic:** Run on any cloud HW (AWS, GCP, Azure, Lambda Labs, On-Prem)


**NOS** inherits its name from **N**itrous **O**xide **S**ystem, the performance-enhancing system typically used in racing cars. NOS is designed to be modular and easy to extend.


## Batteries Included
 - 💪 **SOTA Model Support:** NOS provides out-of-the-box support for popular CV models such as [Stable Diffusion](stabilityai/stable-diffusion-2), [OpenAI CLIP](openai/clip-vit-base-patch32), [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) object detection, tracking and more
 - 🔌 **APIs:** NOS provides out-of-the-box APIs and avoids all the ML model deployment hassles
 - 🐳 **Docker:** NOS ships with docker images to run accelerated and scalable CV workloads
 - 📈 **Multi-Platform**: NOS allows you to run models on different HW (NVIDIA, custom ASICs) without any model compilation or runtime management.


## Getting Started

Get started with the full NOS server by installing via pip:

```shell
conda env create -n nos-py38 python=3.8
conda activate nos-py38
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install torch-nos[server]
```

- [API Documentation](https://autonomi-ai.github.io/nos/)
- [Quickstart](https://autonomi-ai.github.io/nos/docs/quickstart/)


## Contribute
We welcome contributions! Please see our [contributing guide](docs/CONTRIBUTING.md) for more information.
