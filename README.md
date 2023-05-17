# nos 🔥: Nitrous Oxide System (NOS) for Computer Vision

<p align="center">
    <a href="https://pypi.org/project/autonomi-nos/">
        <img alt="PyPi Version" src="https://badge.fury.io/py/autonomi-nos.svg">
    </a>
    <a href="https://pypi.org/project/autonomi-nos/">
        <img alt="PyPi Version" src="https://img.shields.io/pypi/pyversions/autonomi-nos">
    </a>
    <a href="https://pypi.org/project/autonomi-nos/">
        <img alt="PyPi Downloads" src="https://img.shields.io/pypi/dm/autonomi-nos">
    </a>
    <a href="https://discord.gg/QAGgvTuvgg">
        <img alt="Discord" src="https://img.shields.io/badge/discord-chat-purple?color=%235765F2&label=discord&logo=discord">
    </a>
</p>

**NOS** is a PyTorch library for optimizing and running lightning-fast inference of popular computer vision models. NOS inherits its name from "Nitrous Oxide System", the performance-enhancing system typically used in racing cars. NOS is designed to be modular and easy to extend.

## Why NOS?
- ⚡️ **Fast**: NOS is built on top of PyTorch and is designed to run models faster.
- 🔥 **Out-of-the-box Performance**: Run stable diffusion or object detection in under 5 lines, 2-3x faster than vanilla PyTorch.
- 👩‍💻 **Reduce barrier-to-entry**: NOS is designed to be easy to use. No ML optimizations or compilers knowledge necessary.
- 📦 **Modular**: NOS is designed to be modular and easy to extend. Optimize Pytorch models in a few lines of code.
- ⚙️ **HW-accelerated:** NOS is designed to leverage hardware-acceleration down to the metal (GPUs, TPUs, ASICs etc).
- ☁️ **Cloud-agnostic:** NOS is designed to run on any cloud (AWS, GCP, Azure, Lambda Labs, on-prem, etc.).

## Batteries Included
 - 💪 **SOTA Model Support:** NOS comes with support for popular CV models such as Stable Diffusion, ViT, CLIP, and more.
 - 🐳 **Docker:** NOS comes with optimized docker images for accelerated CV workloads (runtime libs, drivers, optimized models).
 - 🔌 **Interfaces:** NOS comes with a REST/gRPC API out-of-the-box to help you use your models.
 - 📈 **Benchmarks**: NOS comes with a suite of benchmarks to help you compare performance of your models.

## Roadmap
We currently plan to support the following hardware:

- GPU (NVIDIA GPUs, AMD GPUs)
    - AWS (g4/g5dn/p3/p4dn)
    - GCP (g2/a1/n1)
- AWS Inferentia inf1/inf2
- Intel Habana Gaudi
- Google TPUv3


## Contribute
We welcome contributions! Please see our [contributing guide](docs/CONTRIBUTING.md) for more information.
