!!!note ""
    **NOS (`torch-nos`)** is a fast and flexible Pytorch inference server, specifically designed for optimizing and running lightning-fast inference of popular foundational AI models.

Optimizing and serving models for production AI inference is still difficult, often leading to notoriously expensive cloud bills and often underutilized GPUs. That’s why we’re building **NOS** - a fast inference server for modern AI workloads. With a few lines of code, developers can optimize, serve, and auto-scale Pytorch model inference without having to deal with the complexities of ML compilers, HW-accelerators, or distributed inference. Simply put, NOS allows AI teams to cut inference costs up to **10x**, speeding up development time and time-to-market.

## ⚡️ Core Features

 - 🔋 **Batteries-included:** Server-side inference with all the necessary batteries (model hub, batching/parallelization, fast I/O, model-caching, model resource management via ModelManager, model optimization via ModelSpec)
 - 📡 **Client-Server architecture:** Multiple lightweight clients can leverage powerful server-side inference workers running remotely without the bloat of GPU libraries, runtimes or 3rd-party libraries.
 - 💪 **High device-utilization:**  With better model management, client’s won’t have to wait on model inference and instead can take advantage of the full GPU resources available. Model multiplexing, and efficient bin-packing of models allow us to leverage the resources optimally (without requiring additional user input).
 - 📦 **Custom model support:** NOS allows you to easily add support for custom models with a few lines of code. We provide a simple API to register custom models with NOS, and allow you to optimize and run models on any hardware (NVIDIA, custom ASICs) without any model compilation or runtime management (see [example](../guides/running-custom-models.md)).


## 👩‍💻 Who is NOS for?

If you've dug this far into the NOS docs (welcome!) you're probably interested in running/serving Pytorch models.

You may have experience with one of the following:

- Deploying to AWS/GCP/Azure/on-prem manually with your own Docker Containers, dependency management etc.
- Using a deployment service like [AWS Sagemaker](https://aws.amazon.com/sagemaker/)
- Hitting an Inference API from OpenAI, Huggingface, Replicate etc.

Each of the above trade off between cost, iteration speed and flexibility. Inference APIs in particular have taken
off for hobbyist developers and enterprises alike that aren't interested in building out their own infra to run Image
Generation, ASR, Object Detection etc. Black-box inference APIs come with drawbacks, however:

- Developers are forced to choose between paying for each and every request even during prototyping, or falling back
to a different execution flow with substantial gaps between dev and prod environments.
- They offer Limited flexibility with regards to model selection/performance optimization. Inference APIs are a fixed quantity.
- They may raise privacy concerns as user data must go outside the wire for inferencing on vendor servers
- Stability issues when using poorly maintained third party APIs

We built **NOS** because we wanted an inference server combining best-practices in model-serving, distributed-inference, auto-scaling all in a single, easy-to-user containerized system that you can simply run with a few lines of Python.

## 📦 Model Containers

Deep learning containers have been around for quite a while, and generally come in the form of a Docker Image
pre-rolled with framework/HW dependencies on top of a base Linux build. More recently, toolchains like Cog
have made wrapping individual model prediction interfaces into containers quick and easy via a DSL, and we expect
this trend to continue. That said, We believe a full-featured Inference Server should be able to do much more, including:

- Serving multiple models dynamically from a single host, eliminating the need for cold starts as workloads change
- Scaling up and down according to inference traffic
- Taking full advantage of HW acceleration and memory optimization to eliminate unnecessary copies for larger input types
 (images/videos)
- Providing a superior developer experience with more error verbosity than 404s from a REST endpoint.

The NOS server / client provide these out of the box with a minimum of installation headache.

## 💬 Give it a try and share your feedback!
NOS is meant to simplify iteration and deployment of popular Generative and Robotics AI workflows. We encourage the
community to give feedback and suggest improvements, and welcome contributions from folks eager to help democratize fast, efficient inference!
