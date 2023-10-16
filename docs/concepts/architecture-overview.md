## Architecture Overview

On a high-level, NOS is a PyTorch library for optimizing and running lightning-fast inference of popular computer vision models. NOS is built to be modular and extensible.

![Unified NOS Inference Server](./assets/arch-how-nos-works.png)

## ⚡️ Core Components

 NOS is built to be modular and extensible. The core components of NOS are:

- [**`ModelManager`**](./model-manager.md): Model manager for serving and running models with Ray actors.
- [**`InferenceService`**](#inferenceservice): Ray-executor based inference service that executes inference requests.
- [**`InferenceRuntimeService`**](#inferenceruntimeservice): Dockerized runtime environment for server-side remote execution

![NOS Architecture](./assets/arch-client-server.png)


### InferenceService

The `InferenceService` along with the `InferenceServiceImpl` gRPC service implementation provides a fully wrapped inference service via gRPC/HTTP2. The `InferenceServiceImpl` wraps the relevant API services such as `ListModels()`, `GetModelInfo()` and crucially `Run()` and executes the inference request via the `InferenceService` class. The `InferenceService` class manages models via the `ModelManager`, and sets up the necessary execution backend via `RayExecutor`. In addition to this, it is also responsible for managing shared memory regions (if requested) for high-performance inference running locally in a single machine.

### Nomenclature

- **Device memory**: We refer to device and GPU memory interchangeably
- **Runtime**: A dockerized runtime environment that has just the pertinent runtime libraries and binaries for execution purposes. The build or compilation libraries are removed via multi-stage builds.
- **Executor**: A single-node ray head that orchestrates inference jobs. We use Ray actors for device-inference and orchestrating auto-scaling (eventually in the multi-node case).
