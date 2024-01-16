---
date: 2024-01-16
tags:
  - infra
  - tools
  - tutorials
categories:
  - infra
  - tutorials
authors:
 - spillai
links:
  - posts/02-nos-tutorials.md
---

# 📚 Getting started with NOS tutorials

We are thrilled to announce a new addition to our resources - the [**NOS Tutorials**](https://github.com/autonomi-ai/nos/tree/main/examples/tutorials)! This series of tutorials is designed to empower users with the knowledge and tools needed to leverage NOS for serving models efficiently and effectively. Whether you're a seasoned developer or just starting out, our tutorials offer insights into various aspects of using NOS, making your journey with model serving a breeze.

Over the next few weeks, we'll walk you through the process of using NOS to serve models, from the basics to more advanced topics. We'll also cover how to use NOS in a production environment, ensuring you have all the tools you need to take your projects to the next level. Finally, keep yourself updated on NOS by giving us a 🌟 on [Github](https://github.com/autonomi-ai/nos).

!!! tip "Can't wait? Show me the code!"
    If you can't wait to get started, head over to our [tutorials](https://github.com/autonomi-ai/nos/tree/main/examples/tutorials) page on [Github](https://github.com/autonomi-ai/nos) to dive right in to the code!

## 🌟 What’s Inside the NOS Tutorials?

The NOS Tutorials encompass a wide range of topics, each focusing on different facets of model serving. Here's a sneak peek into what you can expect:

### 1. Serving custom models: [`01-serving-custom-models`](https://github.com/autonomi-ai/nos/tree/main/examples/tutorials/01-serving-custom-models)
Dive into the world of custom GPU models with NOS. This tutorial shows you how easy it is to wrap your Pytorch code with NOS, and serve them via a REST / gRPC API.

### 2. Serving custom methods: [`02-serving-custom-methods`](https://github.com/autonomi-ai/nos/tree/main/examples/tutorials/02-serving-custom-methods)
Learn how to expose several custom methods of a model for serving. This tutorial is perfect for those looking to tailor their model's functionality to specific requirements, enhancing its utility and performance.

### 3. Serve LLMs with streaming support: [`03-llm-streaming-chat`](https://github.com/autonomi-ai/nos/tree/main/examples/tutorials/03-llm-streaming-chat)
Get hands-on with serving an LLM with streaming support. This tutorial focuses on using [`TinyLlama/TinyLlama-1.1B-Chat-v0.1`](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0), showcasing how to implement streaming capabilities with NOS for smoother, more efficient language model interactions.

### 4. Serve multiple models on the same GPU: [`04-multiple-models`](https://github.com/autonomi-ai/nos/tree/main/examples/tutorials/04-multiple-models)
Step up your game by serving multiple models on the same GPU. This tutorial explores the integration of models like [`TinyLlama/TinyLlama-1.1B-Chat-v0.1`](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0) and [distil-whisper/distil-small.en](https://huggingface.co/distil-whisper/distil-small.en), enabling multi-modal applications such as audio transcription combined with summarization on a single GPU.

### 5. Serving models in production with Docker [`05-serving-with-docker`](https://github.com/autonomi-ai/nos/tree/main/examples/tutorials/05-serving-with-docker)
Enter the realm of production environments with our Docker tutorial. This guide is essential for anyone looking to use NOS in a more structured, scalable environment. You'll learn how to deploy your production NOS images with Docker and Docker Compose, ensuring your model serving works with existing ML infrastructure as reliably as possible.


!!!info "Stay tuned!"
    🔗 **Stay tuned**, as we'll continuously update the section with more tutorials and resources to keep you ahead in the ever-evolving world of model serving!

Happy Model Serving!

---

*This blog post is brought to you by the [NOS Team](https://github.com/autonomi-ai/) - committed to making model serving fast, efficient, and accessible to all!*