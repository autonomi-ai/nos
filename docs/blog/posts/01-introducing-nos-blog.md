---
date: 2024-01-02
tags:
  - infra
  - tools
categories:
  - infra
authors:
 - spillai
 - sloftin
links:
  - posts/01-introducing-nos-blog.md
---

# Introducing NOS Blog!

At [Autonomi AI](https://autonomi.ai), we build infrastructure tools to make AI *fast*, *easy* and *affordable*. 

A few weeks back, we released [NOS](https://github.com/autonomi-ai/nos) - a fast and flexible inference server for [PyTorch](https://pytorch.org/) that can run a whole host of open-source AI models (LLMs, Stable Diffusion, CLIP, Whisper, Object Detection etc) all under one-roof. 

**Today, we’re finally excited to launch the NOS blog**.

We’ve been big believers of *multi-modal* from the very beginning, and you can do all of it with NOS today. Give us a 🌟 on [Github](https://github.com/autonomi-ai/nos) if you're stoked -- NOS can run locally on your Linux desktop (with a gaming GPU), in any cloud GPU (NVIDIA L4, A100s, etc) and even on CPUs (without any acceleration). Very soon, we'll support running models on [Apple Silicon](https://www.apple.com/newsroom/2023/10/apple-unveils-m3-m3-pro-and-m3-max-the-most-advanced-chips-for-a-personal-computer/) and custom AI accelerators such as [Inferentia2](https://aws.amazon.com/ec2/instance-types/inf2/) from Amazon Web Services (AWS).


## 🎯 Why are we building yet another AI inference server?

Most inference API implementations today deeply couple the API framework ([FastAPI](https://fastapi.tiangolo.com/), [Flask](https://flask.palletsprojects.com/en/3.0.x/)) with the modeling backend ([PyTorch](https://pytorch.org/), [TF](https://www.tensorflow.org/) etc) - in other words, it doesn’t let you separate the concerns for the AI backend (e.g. AI hardware, drivers, model compilation, execution runtime, scale out, memory efficiency, async/batched execution, multi-model management etc) from your AI application (e.g. auth, observability, telemetry, web integrations etc), especially if you’re looking to build a production-ready application.

We’ve made it very easy for developers to host new PyTorch models as APIs and take them to production without having to worry about any of the backend infrastructure concerns. We build on some awesome projects like [FastAPI](https://fastapi.tiangolo.com/), [Ray](https://ray.io/), [Hugging Face](https://www.huggingface.co), [transformers](https://github.com/huggingface/transformers) and [diffusers](https://github.com/huggingface/transformers).


!!! info "What's coming?"
    Over the coming weeks, we’ll be announcing some *awesome features* that we believe will make the power of large foundation models more accessible, cheaper and easy-to-use than ever before. 

## 🥜 NOS, in a nutshell

NOS was built from the ground-up, with developers in mind. Here are a few things we think developers care about:

- 🥷 **Flexible**: Support for OSS models with custom runtimes with pip, conda and cuda/driver dependencies.
- 🔌 **Pluggable:** Simple API over a high-performance gRPC or REST API that supports batched requests, and streaming.
- 📦 **Extensible**: Written entirely in Python so it’s easily hackable and extensible with an Apache-2.0 License for commercial use.
- 🚀 **Scalable**: Serve multiple custom models simultaneously on a single or multi-GPU instance, without having to worry about memory management and model scaling.
- 🏛️ **Local**: Local execution means that you control your data, and you’re free to build NOS for domains that are more restrictive with data-privacy.
- ☁️ **Cloud-agnostic:** Fully containerized means that you can develop, test and deploy NOS locally, on-prem, on any cloud or AI CSP.

Go ahead and check out our [playground](https://github.com/autonomi-ai/nos-playground), and try out some of the more recent models with NOS.

## 🔗 Relevant Links

- ⭐️ Github: [https://github.com/autonomi-ai/nos](https://github.com/autonomi-ai/nos)
- 👩‍💻 Playground: [https://github.com/autonomi-ai/nos-playground](https://github.com/autonomi-ai/nos-playground)
- 📚 Docs: [https://docs.nos.run/](https://docs.nos.run/)
- 💬 [Discord](https://discord.gg/QAGgvTuvgg), [X / Twitter](https://twitter.com/autonomi_ai), [LinkedIn](https://www.linkedin.com/company/autonomi-ai/)

