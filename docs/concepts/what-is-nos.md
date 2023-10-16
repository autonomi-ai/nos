!!!note ""
    **NOS** is a PyTorch library for optimizing and running lightning-fast inference of popular computer vision models.

Optimizing and serving models for production AI inference is still difficult, often leading to notoriously expensive cloud bills and often underutilized GPUs. That’s why we’re building **NOS** - a flexible and performance-tunable inference server for running modern AI workloads. With a few lines of code, developers can optimize, serve, and auto-scale Pytorch model inference without having to deal with the complexities of ML compilers, HW-accelerators, or distributed inference. Simply put, NOS allows AI teams to cut inference costs up to **10x**, speeding up development time and time-to-market.

## Core Features
 - 🔋 **Batteries-included:** Server-side inference with all the necessary batteries (model hub, batching/parallelization, fast I/O, model-caching, model resource management via ModelManager, model optimization via ModelSpec)
 - 📡 **Client-Server architecture:** Multiple lightweight clients can leverage powerful server-side inference workers running remotely without the bloat of GPU libraries, runtimes or 3rd-party libraries.
 - 💪 **High device-utilization:**  With better model management, client’s won’t have to wait on model inference and instead can take advantage of the full GPU resources available. Model multiplexing, and efficient bin-packing of models allow us to leverage the resources optimally (without requiring additional user input).
 - 📦 **Custom model support:** NOS allows you to easily add support for custom models with a few lines of code. We provide a simple API to register custom models with NOS, and allow you to optimize and run models on any hardware (NVIDIA, custom ASICs) without any model compilation or runtime management (see [example](../guides/running-custom-models.md)).
