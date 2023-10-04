# Supported Runtimes

The following runtimes are supported by NOS.

| Status | Name | Pyorch | HW | Base | Description |
| - | --- | --- | --- | --- | --- |
| ✅ | `cpu` | [`2.0.1`](https://pypi.org/project/torch/2.0.1/) | CPU | `python:3.8.10-slim` | CPU-only runtime. |
| ✅ | `gpu` | [`2.0.1`](https://pypi.org/project/torch/2.0.1/) | GPU | `nvidia/cuda:11.8.0-base-ubuntu22.04` | GPU runtime. |
| ✅ | `trt` | [`2.0.1`](https://pypi.org/project/torch/2.0.1/) | GPU | `nvidia/cuda:11.7.0-base-ubuntu22.04` | GPU runtime with TensorRT (8.4.2.4). |
| ✅ | `inf2` | [`1.13.1`](https://pypi.org/project/torch/1.13.1/) | [AWS Inferentia2](https://aws.amazon.com/ec2/instance-types/inf2/) | [`autonomi/nos:latest-cpu`](https://hub.docker.com/r/autonomi/nos/tags) | Inf2 runtime with [torch-neuronx](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/frameworks/torch/torch-neuronx/setup/pytorch-install.html). |
