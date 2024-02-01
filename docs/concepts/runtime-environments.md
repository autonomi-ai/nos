The NOS inference server supports custom runtime environments through the use of the `InferenceServiceRuntime` class and the [configurations](../api/server.md#InferenceServiceRuntime.configs) defined within. This class provides a high-level interface for defining new custom runtime environments that can be used with NOS.

### ⚡️ NOS Inference Runtime

We use docker to configure different worker configurations to run workloads in different runtime environments. The configured runtime environments are specified in the [InferenceServiceRuntime](../api/server.md#inferenceserviceruntime) class, which wraps the generic [`DockerRuntime`] class. For convenience, we have pre-built some runtime environments that can be used out-of-the-box `cpu`, `gpu`, `inf2` etc.

This is the general flow of how the runtime environments are configured:
- Configure runtime environments including `cpu`, `gpu`, `inf2` etc in the [`InferenceServiceRuntime`](../api/server.md#inferenceserviceruntime) `config` dictionary.
- Start the server with the appropriate runtime environment via the `--runtime` flag.
- The ray cluster is now configured within the appropriate runtime environment and has access to the appropriate libraries and binaries.

For custom runtime support, we use [Ray](https://ray.io) to configure different worker configurations (custom conda environment, with resource naming) to run workers on different runtime environments (see below).

### 🏃‍♂️ Supported Runtimes

The following runtimes are supported by NOS:

| Status | Name | Pyorch | HW | Base | Size | Description |
| - | --- | --- | --- | --- | --- | --- |
| ✅ | [`autonomi/nos:latest-cpu`](https://hub.docker.com/r/autonomi/nos/tags)  | [`2.1.1`](https://pypi.org/project/torch/2.1.1/) | CPU | `debian:buster-slim` | 1.1 GB | CPU-only runtime. |
| ✅ | [`autonomi/nos:latest-gpu`](https://hub.docker.com/r/autonomi/nos/tags)  | [`2.1.1`](https://pypi.org/project/torch/2.1.1/) | NVIDIA GPU | `nvidia/cuda:11.8.0-base-ubuntu22.04` | 3.9 GB | GPU runtime. |
| ✅ | [`autonomi/nos:latest-inf2`](https://hub.docker.com/r/autonomi/nos/tags) | [`1.13.1`](https://pypi.org/project/torch/1.13.1/) | [AWS Inferentia2](https://aws.amazon.com/ec2/instance-types/inf2/) | `debian:buster-slim` | 1.7 GB | Inf2 runtime with [torch-neuronx](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/frameworks/torch/torch-neuronx/setup/pytorch-install.html). |
| **Coming Soon** | `trt` | [`2.0.1`](https://pypi.org/project/torch/2.0.1/) | NVIDIA GPU | `nvidia/cuda:11.8.0-base-ubuntu22.04` | GPU runtime with TensorRT (8.4.2.4). |

### 🛠️ Adding a custom runtime

To define a new custom runtime environment, you can extend the `InferenceServiceRuntime` class and add new configurations to the existing `configs` variable.

::: nos.server._runtime.InferenceServiceRuntime.configs
