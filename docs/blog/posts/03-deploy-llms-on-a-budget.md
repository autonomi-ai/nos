---
date: 2024-01-18
tags:
  - llm
  - integrations
  - tutorials
  - budget
categories:
  - infra
  - tutorials
authors:
 - spillai
links:
  - posts/03-deploy-llms-on-a-budget.md
---

# Serving LLMs for less than $160 / month

<img src="/docs/blog/assets/nos-phixtral.jpg" width="100%">

Deploying Large Language Models (LLMs) and Mixture of Experts (MoEs) are all the rage today, and for good reason. They are the most powerful and [closest](https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard) open-source models in terms of performance to OpenAI GPT-3.5 today. However, it turns out that deploying these models can still be somewhat of a lift for most ML engineers and researchers, both in terms of engineering work and operational costs. For example, the recently announced [Mixtral 8x7B](https://mistral.ai/news/mixtral-of-experts/) requires 2x [NVIDIA A100-80G GPUs](https://www.nvidia.com/en-us/data-center/a100/), which can cost upwards of $5000 / month (on-demand) on CSPs.

With recent advancements in [model compression](https://huggingface.co/docs/optimum/intel/optimization_inc#optimization), [quantization](https://github.com/mit-han-lab/llm-awq) and [model mixing](https://mistral.ai/news/mixtral-of-experts/), we are now seeing an exciting race unfold to deploy these expert models on a budget, without sacrificing significantly on performance. In this blog post, we'll show you how to deploy the [mlabonne/phixtral-4x2_8](https://huggingface.co/mlabonne/phixtral-4x2_8) model on a single [NVIDIA L4 GPU](https://www.nvidia.com/en-us/data-center/l4/) for under $160 / month and easily scale-out a dirt-cheap, dedicated inference service of your own. We'll be using [SkyPilot](https://github.com/skypilot-org/skypilot) to deploy and manage our [NOS](https://github.com/autonomi-ai/nos) service on spot (pre-emptible) instances, making them especially cost-efficient.

## 🧠 What is Phixtral?

Inspired inspired by the [mistralai/Mixtral-8x7B-v0.1](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1) architecture, [mlabonne/phixtral-4x2_8](https://huggingface.co/mlabonne/phixtral-4x2_8) is the first Mixure of Experts (MoE) made with 4 [microsoft/phi-2](https://huggingface.co/microsoft/phi-2) models that was recently MIT licensed. The general idea behind mixture-of-experts is to combine the capabilities of multiple models to achieve better performance than each individual model. They are [significantly more memory-efficient](https://huggingface.co/blog/moe#what-is-a-mixture-of-experts-moe) for inference too, but that's a post for a later date. In this case, we combine the capabilities of 4 [microsoft/phi-2](https://huggingface.co/microsoft/phi-2) models to achieve better performance than each of the individual 2.7B parameter models it's composed of. 

??? note "Breakdown of the [mlabonne/phixtral-4x2_8](https://huggingface.co/mlabonne/phixtral-4x2_8) model"

    Here's the breakdown of the 4 models that make up the [mlabonne/phixtral-4x2_8](https://huggingface.co/mlabonne/phixtral-4x2_8) model:

    ```yaml
    base_model: cognitivecomputations/dolphin-2_6-phi-2
    gate_mode: cheap_embed
    experts:
      - source_model: cognitivecomputations/dolphin-2_6-phi-2
        positive_prompts: [""]
      - source_model: lxuechen/phi-2-dpo
        positive_prompts: [""]
      - source_model: Yhyu13/phi-2-sft-dpo-gpt4_en-ep1
        positive_prompts: [""]
      - source_model: mrm8488/phi-2-coder
        positive_prompts: [""]
    ```

    You can go to the original model card [here](https://huggingface.co/mlabonne/phixtral-4x2_8) for more details on how the model was merged using [mergekit](https://github.com/cg123/mergekit).

Now, let's take a look at the performance of the [mlabonne/phixtral-4x2_8](https://huggingface.co/mlabonne/phixtral-4x2_8) model on the [Nous Suite](https://github.com/mlabonne/llm-autoeval?tab=readme-ov-file#evaluation-parameters) compared to other models in the 2.7B parameter range. 

| Model | AGIEval	| GPT4All	| TruthfulQA	| Bigbench	| Average | 
| --- | --- | --- | --- | --- | --- |
| **[mlabonne/phixtral-4x2_8](https://huggingface.co/mlabonne/phixtral-4x2_8)** | **33.91** | 70.44 | 48.78 | **37.82** | **47.78** |
| [dolphin-2_6-phi-2](https://huggingface.co/cognitivecomputations/dolphin-2_6-phi-2) | 33.12 | 69.85 | 47.39 | 37.2 | 46.89 |
| [phi-2-dpo](https://huggingface.co/lxuechen/phi-2-dpo) | 30.39 | **71.68** | **50.75** | 34.9 | 46.93 |
| [phi-2](https://huggingface.co/microsoft/phi-2) | 27.98 | 70.8 | 44.43 | 35.21 | 44.61 |


## 💸 Serving Phixtral on a budget with SkyPilot and NOS

Let's now see how we can deploy the [mlabonne/phixtral-4x2_8](https://huggingface.co/mlabonne/phixtral-4x2_8) model on a single NVIDIA L4 GPU for under $160 / month. We'll be using [SkyPilot](https://github.com/skypilot-org/skypilot) to deploy and manage our [NOS](https://github.com/autonomi-ai/nos) service on spot (pre-emptible) instances, making them especially cost-efficient.

!!!question "What's SkyPilot?"
    If you're new to SkyPilot, we recommend you go through our [NOS x SkyPilot integration page](/docs/integrations/skypilot.html){:target="_blank"} first to familiarize yourself with the tool.

### 1. Define your custom model and serve specification

In this example, we'll be using the [`llm-streaming-chat`](https://github.com/autonomi-ai/nos-playground/blob/main/examples/llm-streaming-chat/) tutorial on [NOS playground](https://github.com/autonomi-ai/nos-playground). First, we'll define our custom phixtral chat model [`phixtral_chat.py`](https://github.com/autonomi-ai/nos-playground/blob/main/examples/llm-streaming-chat/models/phixtral_chat.py) and a [`serve.phixtral.yaml`](https://github.com/autonomi-ai/nos-playground/blob/main/examples/llm-streaming-chat/serve.phixtral.yaml) serve specification that will be used by NOS to serve our model. The relevant files are shown below:

```bash
(nos-py38) nos-playground/examples/llm-streaming-chat $ tree .
├── models
│   └── phixtral_chat.py
├── serve.phixtral.yaml
```

The entire chat interface is defined in the `StreamingChat` module in [`phixtral_chat.py`](https://github.com/autonomi-ai/nos-playground/blob/main/examples/llm-streaming-chat/models/phixtral_chat.py), where the `chat` method returns a string iterable for the gRPC / HTTP server to stream back model predictions to the client. 

The [`serve.phixtral.yaml`](https://github.com/autonomi-ai/nos-playground/blob/main/examples/llm-streaming-chat/serve.phixtral.yaml) serve specification defines the custom chat model, and a custom runtime that NOS uses to execute our model. Follow the annotations below to understand the different components of the serve specification.

<div class="annotate" markdown>

```yaml title="serve.phixtral.yaml"

images: (1)
  llm-py310-cu121: (2)
    base: autonomi/nos:latest-py310-cu121 (3)
    pip: (4)
      - bitsandbytes
      - transformers
      - einops
      - accelerate

models: (5)
  mlabonne/phixtral-4x2_8: (6)
    model_cls: StreamingChat (7)
    model_path: models/phixtral_chat.py (8)
    init_kwargs:
      model_name: mlabonne/phixtral-4x2_8
    default_method: chat
    runtime_env: llm-gpu
    deployment: (9)
      resources:
        device: auto
        device_memory: 7Gi (10)
```
</div>
1. Specifies the custom runtime images that will be used to serve our model.
2. Specifies the name of the custom runtime image (referenced below in `runtime_env`). 
3. Specifies the base NOS image to use for the custom runtime image. We provide a few pre-built images on [dockerhub](https://hub.docker.com/repository/docker/autonomi/nos/general). 
4. Specifies the pip dependencies to install in the custom runtime image.
5. Specifies all the custom models we intend to serve.
6. Specifies the unique name of the custom model (model identifier).
7. Specifies the model class to use for the custom model.
8. Specifies the path to the model class definition.
9. Specifies the deployment resources needed for the custom model.
10. Specifies the GPU memory to allocate for the custom model.

### 2. Test your custom model locally with NOS

In order to start the NOS server locally, we can simply run the following command:

```bash
nos serve up -c serve.phixtral.yaml --http
```

This will build the custom runtime image, and start the NOS server locally, exposing an OpenAI compatible HTTP proxy on port `8000`. This will allow you to chat with your custom LLM endpoint using any OpenAI API compatible client. 

### 3. Deploy your NOS service with SkyPilot

Now that we have defined our serve YAML specification, let's deploy this service on GCP using [SkyPilot](https://github.com/skypilot-org/skypilot). In this example, we're going to use SkyPilot's `sky serve` command to deploy our NOS service on spot (pre-emptible) instances on GCP. 

??? note "Deploy on any cloud provider (AWS, Azure, GCP, OCI, Lambda Labs, etc.)"
    SkyPilot supports deploying NOS services on any cloud provider. In this example, we're going to use GCP, but you can easily deploy on AWS, Azure, or any other cloud provider of your choice. You can override `gcp` by providing the `--cloud` flag to `sky serve up`.

Let's define a serving configuration in a [`service-phixtral.sky.yaml`](https://github.com/autonomi-ai/nos-playground/blob/main/deployments/deploy-llms-with-skypilot/service-phixtral.sky.yaml) file. This YAML specification will be used by SkyPilot to deploy and manage our NOS service on pre-emptible instances, automatically provisioning and recovering from failovers, setting up new instances when needed on server pre-emptions. 

<div class="annotate" markdown>

```yaml title="service-phixtral.sky.yaml"

name: service-phixtral

file_mounts:
  /app: ./app (1)

resources:
  cloud: gcp
  accelerators: L4:1
  use_spot: True (2)
  ports:
    - 8000

service:
  readiness_probe: (3)
    path: /v1/health 
  replicas: 2 (4)

setup: |
  sudo apt-get install -y docker-compose-plugin
  pip install torch-nos

run: |
  cd /app && nos serve up -c serve.phixtral.yaml --http (5)

```

</div>
1. Setup file-mounts to mount the local `./app` directory so that the `serve.phixtral.yaml` and `models/` directory are available on the remote instance.
2. Use spot (pre-emptible) instances instead of on-demand instances.
3. Define the readiness probe path for the service. This allows the SkyPilot controller to check the health of the service and recover from failures if needed.
4. Define the number of replicas to deploy.
5. Define the `run` command to execute on each replica. In this case, we're simply starting the NOS server with the phixtral model deployed on init. 

To deploy our NOS service, we can simply run the following command:

```bash
sky serve up -n service-mixtral service-mixtral.sky.yaml
```

SkyPilot will automatically pick the cheapest region and zone to deploy our service, and provision the necessary cloud resources to deploy the NOS server. In this case, you'll notice that SkyPilot provisioned two [NVIDIA L4 GPU](https://www.nvidia.com/en-us/data-center/l4/) instances on GCP in the `us-central1-a` availability zone. 

You should see the following output:

```bash
(nos-infra-py38) deployments/deploy-llms-with-skypilot $ sky serve up -n service-mixtral service-mixtral.sky.yaml
Service from YAML spec: service-mixtral.sky.yaml
Service Spec:
Readiness probe method:           GET /v1/health
Readiness initial delay seconds:  1200
Replica autoscaling policy:       Fixed 2 replicas
Replica auto restart:             True
Each replica will use the following resources (estimated):
I 01-19 16:01:58 optimizer.py:694] == Optimizer ==
I 01-19 16:01:58 optimizer.py:705] Target: minimizing cost
I 01-19 16:01:58 optimizer.py:717] Estimated cost: $0.2 / hour
I 01-19 16:01:58 optimizer.py:717]
I 01-19 16:01:58 optimizer.py:840] Considered resources (1 node):
I 01-19 16:01:58 optimizer.py:910] ----------------------------------------------------------------------------------------------------
I 01-19 16:01:58 optimizer.py:910]  CLOUD   INSTANCE              vCPUs   Mem(GB)   ACCELERATORS   REGION/ZONE     COST ($)   CHOSEN
I 01-19 16:01:58 optimizer.py:910] ----------------------------------------------------------------------------------------------------
I 01-19 16:01:58 optimizer.py:910]  GCP     g2-standard-4[Spot]   4       16        L4:1           us-central1-a   0.22          ✔
I 01-19 16:01:58 optimizer.py:910] ----------------------------------------------------------------------------------------------------
I 01-19 16:01:58 optimizer.py:910]
Launching a new service 'service-mixtral'. Proceed? [Y/n]: y
Launching controller for 'service-mixtral'
...
I 01-19 16:02:14 cloud_vm_ray_backend.py:1912] Launching on GCP us-west1 (us-west1-a)
I 01-19 16:02:30 log_utils.py:45] Head node is up.
I 01-19 16:03:03 cloud_vm_ray_backend.py:1717] Successfully provisioned or found existing VM.
I 01-19 16:03:05 cloud_vm_ray_backend.py:4558] Processing file mounts.
...
I 01-19 16:03:20 cloud_vm_ray_backend.py:3325] Setup completed.
I 01-19 16:03:29 cloud_vm_ray_backend.py:3422] Job submitted with Job ID: 11

Service name: service-mixtral
Endpoint URL: XX.XXX.X.XXX:30001
To see detailed info:           sky serve status service-mixtral [--endpoint]
To teardown the service:        sky serve down service-mixtral

To see logs of a replica:       sky serve logs service-mixtral [REPLICA_ID]
To see logs of load balancer:   sky serve logs --load-balancer service-mixtral
To see logs of controller:      sky serve logs --controller service-mixtral

To monitor replica status:      watch -n10 sky serve status service-mixtral
To send a test request:         curl -L XX.XX.X.XXX:30001
```

Once the service is deployed, you can get the IP address of the SkyPilot service via:. 

```bash
sky serve status service-phixtral --endpoint
```

We'll refer to `<sky-serve-ip>` as the load balancer's IP address, that takes the full form of `<sky-serve-ip>:30001`. You should now be able to ping the load-balancer endpoint directly with `cURL` and see the following output:

```bash
$ curl -L http://<sky-serve-ip>:30001/v1/health
{"status":"ok"}
```

## 💬 Chatting with your custom Phixtral service

You're now ready to chat with your hosted, custom LLM endpoint! Here's a quick demo of the [mlabonne/phixtral-4x2_8](https://huggingface.co/mlabonne/phixtral-4x2_8) model served with NOS across 2 spot (pre-emptible) instances. 

<script async id="asciicast-632285" src="https://asciinema.org/a/632285.js"></script>

On the top, you'll see the logs from both the serve replicas, and the corresponding chats that are happening *concurrently* on the bottom. SkyPilot handles the load-balancing and routing of requests to the replicas, while NOS handles the custom model serving and streaming inference. Below, we show you how you can chat with your hosted LLM endpoint using `cURL`, an OpenAI compatible client, or the OpenAI Python client.


=== "Using cURL"
  
    ```bash
    curl \
    -X POST -L http://<sky-serve-ip>:30001/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "mlabonne/phixtral-4x2_8",
        "messages": [{"role": "user", "content": "Tell me a joke in 300 words"}],
        "temperature": 0.7, "stream": true
      }'
    ```

=== "Using an OpenAI compatible client"

    Below, we show how you can use any OpenAI API compatible client to chat with your hosted LLM endpoint. We will use the popular [llm](https://github.com/simonw/llm) CLI tool from [Simon Willison](https://simonwillison.net/) to chat with our hosted LLM endpoint.

    ```bash
    # Install the llm CLI tool
    $ pip install llm

    # Install the llm-nosrun plugin to talk to your service
    $ llm install llm-nosrun

    # List the models
    $ llm models list

    # Chat with your endpoint
    $ NOSRUN_API_BASE=http://<sky-serve-ip>:30001/v1 llm -m mlabonne/phixtral-4x2_8 "Tell me a joke in 300 words"
    ```

=== "Using the OpenAI Python client"

    Below, we show how you can use the [OpenAI Python Client](https://github.com/openai/openai-python) to chat with your hosted LLM endpoint.

    ```python
    import openai

    # Create a stream and print the output
    client = openai.OpenAI(api_key="no-key-required", base_url=f"http://<sky-serve-ip>:30001/v1")
    stream = client.chat.completions.create(
        model="mlabonne/phixtral-4x2_8",
        messages=[{"role": "user", "content": "Tell me a joke in 300 words"}],
        stream=True,
    )
    for chunk in stream:
        print(chunk.choices[0].delta.content or "", end="")
    ```


!!!info
    In these examples, we use SkyPilot's load-balancer port `30001`  which redirects HTTP traffic to one of the many NOS replicas (on port `8000`) in a round-robin fashion. This allows us to scale-out our service to multiple replicas, and load-balance requests across them.

## 🤑 What's it going to cost me?

In the example above, we were able to deploy the [mlabonne/phixtral-4x2_8](https://huggingface.co/mlabonne/phixtral-4x2_8) model on a single [NVIDIA L4 GPU](https://www.nvidia.com/en-us/data-center/l4/) for ~**$0.22** / hour / replica, or ~**$160** / month / replica**. This is a ~**45x** improvement over the cost of deploying the [mistralai/Mixtral-8x7B-v0.1](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1) model on 2x [NVIDIA A100-80G GPUs](https://www.nvidia.com/en-us/data-center/a100/), which can cost upwards of $7000 / month (on-demand) on CSPs. As advancements in model compression, quantization and model mixing continue to improve, we expect more users to be able to fine-tune, distill and deploy these expert [small-langauge models](https://www.microsoft.com/en-us/research/blog/orca-2-teaching-small-language-models-how-to-reason/) on a budget, without sacrificing significantly on performance.

The table below shows the costs of deploying one of these popular MoE LLM models on a single GPU server on GCP. As you can see, the cost of deploying a single model can range from $500 to $7300 / month, depending on the model and of course CSP (kept fixed here). 


| Model | Cloud Provider | GPU | VRAM | Spot | Cost / hr | Cost / month |
| --- | --- | --- | --- | --- | --- | --- |
| [mistralai/Mixtral-8x7B-v0.1](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1) | GCP | 2x NVIDIA A100-80G | ~94 GB | - | $10.05 | ~$7236 |
| [TheBloke/Mixtral-8x7B-Instruct-v0.1-AWQ](https://huggingface.co/TheBloke/Mixtral-8x7B-Instruct-v0.1-AWQ) | GCP | NVIDIA A100-40G | ~25GB | - | $3.67 | ~$2680 |
| [mlabonne/phixtral-4x2_8](https://huggingface.co/mlabonne/phixtral-4x2_8) | GCP | NVIDIA L4 | ~9GB | - | $0.70 | ~$500 |
| **[mlabonne/phixtral-4x2_8](https://huggingface.co/mlabonne/phixtral-4x2_8)** | GCP | NVIDIA L4 | ~9GB | ✅ | **$0.22** | **~$160** |

However, the **onus is on the developer** to figure out the **right instance type, spot instance strategy, and the right number of replicas** to deploy to ensure that the service is both *cost-efficient and performant*. In the coming weeks, we're going to be introducing some exciting tools to help developers alleviate this pain and provide transparency to make the right infrastructure decisions for their services. Stay tuned!

## 🎁 Wrapping up

In this blog post, we showed you how to deploy the [mlabonne/phixtral-4x2_8](https://huggingface.co/mlabonne/phixtral-4x2_8) model on a single NVIDIA L4 GPU for under $160 / month and scale-out a dirt-cheap inference service of your own. We used [SkyPilot](https://github.com/skypilot-org/skypilot) to deploy and manage our [NOS](https://github.com/autonomi-ai/nos) service on spot (pre-emptible) instances, making them especially cost-efficient.

In our next blog post, we’ll take it one step further. We'll explore how you can serve multiple models on the same GPU so that **your infrastructure costs don’t have to scale with the number of models you serve**. The **TL;DR** is that you will soon be able to serve multiple models with fixed and predictable pricing, making model serving more accessible and cost-efficient than ever before.

<br>