In NOS, the [`ModelSpec`](../api/common/spec.html#nos.common.spec.ModelSpec) class is a serializable specification of a model that captures all the relevant information for instantatiation, execution and runtime profile of a model.


## Model Specification
- **Deterministic:** We benchmark the models during model registry, so you are guaranteed execution runtimes and device resource-usage. More specifically, the model specification will allow us to measure memory consumption and FLOPs ahead-of-time and enable more efficient device-memory usage in production.
- **Scalable:** Registered models can be independently scaled up for batch inference or parallel execution with Ray actors.
- **Optimizable:** Every registered model can be inspected, compiled and optimized with a unique and configurable runtime-engine ([TensorRT](https://developer.nvidia.com/tensorrt), [ONNX](https://onnxruntime.ai/), [AITemplate](https://github.com/facebookincubator/AITemplate) etc). This allows us to benchmark models before they enter production, and run models at the optimal (or configurable) operating point.
