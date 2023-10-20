## nos.server

## Docker Runtime

The docker runtime provides the `docker-py` interface to run containerized inference workloads using Docker. It allows starting and stopping containers, getting container information, and running the containers programmatically with HW support for accelerators like GPUs, ASICs etc. 

::: nos.server._docker.DeviceRequest

::: nos.server._docker.DockerRuntime

---
## InferenceServiceRuntime

::: nos.server._runtime.InferenceServiceRuntime

## InferenceService

The `InferenceService` along with the `InferenceServiceImpl` gRPC service implementation provides a fully wrapped inference service via gRPC/HTTP2. The `InferenceServiceImpl` wraps the relevant API services such as `ListModels()`, `GetModelInfo()` and crucially `Run()` and executes the inference request via the `InferenceService` class. The `InferenceService` class manages models via the `ModelManager`, and sets up the necessary execution backend via `RayExecutor`. In addition to this, it is also responsible for managing shared memory regions (if requested) for high-performance inference running locally in a single machine.

::: nos.server._service.ModelHandle

::: nos.server._service.InferenceServiceImpl

::: nos.server._service.serve
