---
date: 2024-02-01
tags:
  - embeddings
  - integrations
  - inferentia
categories:
  - infra
  - embeddings
  - asic
authors:
 - spillai
links:
  - posts/04-introducing-inf2-runtime.md
---

# Introducing the NOS Inferentia2 Runtime

<img src="/docs/blog/assets/nos-inf2.jpg" width="100%">

We are excited to announce the availability of the [AWS Inferentia2](https://aws.amazon.com/en/ec2/instance-types/inf2/) runtime on [NOS](https://github.com/autonomi-ai/nos) - a.k.a. our **[`inf2`](../../concepts/runtime-environments.md#🏃‍♂️-supported-runtimes)** runtime. This runtime is designed to easily serve models on AWS Inferentia2, a high-performance, purpose-built chip for inference. In this blog post, we will introduce the AWS Inferentia2 runtime, and show you how to trivially deploy a model on the AWS Inferentia2 device using NOS. If you have followed our previous tutorial on [serving LLMs on a budget (on NVIDIA hardware)](./03-serving-llms-on-a-budget.md), you will be pleasantly surprised to see how easy it is to deploy a model on the AWS Inferentia2 device using the pre-baked NOS **[`inf2`](../../concepts/runtime-environments.md#🏃‍♂️-supported-runtimes)** runtime we provide.

## ⚡️ What is AWS Inferentia2?

[AWS Inferentia2](https://aws.amazon.com/en/ec2/instance-types/inf2/) (Inf2 for short) is the second-generation inference accelerator from AWS. Inf2 instances raise the performance of [Inf1](https://aws.amazon.com/ec2/instance-types/inf1/) (originally launched in 2019) by delivering 3x higher compute performance, 4x larger total accelerator memory, up to 4x higher throughput, and up to 10x lower latency. Inf2 instances are the first inference-optimized instances in Amazon EC2 to support scale-out distributed inference with ultra-high-speed connectivity between accelerators. 

Relative to the [AWS G5 instances](https://aws.amazon.com/ec2/instance-types/g5/) ([NVIDIA A10G](https://www.nvidia.com/en-us/data-center/products/a10-gpu/)), Inf2 instances promise up to 50% better performance-per-watt. Inf2 instances are ideal for applications such as natural language processing, recommender systems, image classification and recognition, speech recognition, and language translation that can take advantage of scale-out distributed inference. 

| Instance Size | Inf2 Accelerators | Accelerator Memory (GB) | vCPU | Memory (GiB) | On-Demand Price | Spot Price  |
|---------------|-------------------|-------------------------|------|--------------|-----------------|-------------|
| inf2.xlarge   | 1                 | 32                      | 4    | 16           | $0.76           | $0.32       |
| inf2.8xlarge  | 1                 | 32                      | 32   | 128          | $1.97           | $0.79       |
| inf2.24xlarge | 6                 | 192                     | 96   | 384          | $6.49           | $2.45       |
| inf2.48xlarge | 12                | 384                     | 192  | 768          | $12.98          | $5.13       |

## 🏃‍♂️ NOS Inference Runtime

The NOS inference server supports custom runtime environments through the use of the [InferenceServiceRuntime](../api/server.md#inferenceserviceruntime) class - a high-level interface for defining new **containerized** and **hardware-aware** runtime environments. NOS already ships with [runtime environments](../../concepts/runtime-environments.md#🏃‍♂️-supported-runtimes) for NVIDIA GPUs (`gpu`) and Intel/ARM CPUs (`cpu`). Today, we're adding the [NOS Inferentia2 runtime](https://hub.docker.com/repository/docker/autonomi/nos/general) (`inf2`) with the [AWS Neuron drivers](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/setup/neuron-setup/pytorch/neuronx/ubuntu/torch-neuronx-ubuntu22.html#setup-torch-neuronx-ubuntu22), the [AWS Neuron SDK](https://github.com/aws-neuron/aws-neuron-sdk) and NOS pre-installed. This allows developers to quickly develop applications for AWS Inferentia2, without wasting any precious time on the complexities of setting up the AWS Neuron SDK and the AWS Inferentia2 driver environments.

## 📦 Deploying a PyTorch model on Inferentia2 with NOS

Deploying PyTorch models on AWS Inferentia2 chips presents a unique set of challenges, distinct from the experience with NVIDIA GPUs. This is primarily due to the static graph execution requirement of ASICs, requiring the user to [trace](https://pytorch.org/docs/stable/generated/torch.jit.trace.html) and [compile models ahead-of-time](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html), making them less accessible to entry-level developers. In some cases, custom model tracing and compilation are essential steps to fully utilize the AWS Inferentia2 accelerators. This demands a deep understanding of the HW-specific deployment/compiler toolchain ([TensorRT](https://developer.nvidia.com/tensorrt), [AWS Neuron SDK](https://github.com/aws-neuron/aws-neuron-sdk)), the captured and data-dependent traced PyTorch graph, and the underlying HW-specific kernel/op-support to name just a few challenges. 

!!!tip "**Simplifying AI hardware access with NOS**"
    NOS aims to bridge this gap and streamline the deployment process, making it more even accessible for both entry-level and expert developers to leverage the powerful inference capabilities of AWS Inferentia2 for their inference needs. 


### 1. Define your custom `inf2` model

In this example, we'll be using the [`inf2/embeddings`](https://github.com/autonomi-ai/nos/blob/main/examples/inf2/embeddings/) sentence embedding tutorial on [NOS](https://github.com/autonomi-ai/nos). First, we'll define our custom [`EmbeddingServiceInf2`](https://github.com/autonomi-ai/nos/blob/main/examples/inf2/embeddings/models/embeddings_inf2.py#L24) model [`models/embeddings_inf2.py`](https://github.com/autonomi-ai/nos/blob/main/examples/inf2/embeddings/models/embeddings_inf2.py) and a [`serve.yaml`](https://github.com/autonomi-ai/nos/blob/main/examples/inf2/embeddings/serve.yaml) serve specification that will be used by NOS to serve our model on the AWS Inferentia2 device. The relevant files are shown below:

<div class="annotate" markdown>

```title="Directory structure of <code>nos/examples/inf2/embeddings</code>"
$ tree .
├── job-inf2-embeddings-deployment.yaml
├── models
│   └── embeddings_inf2.py  (1)
├── README.md
├── serve.yaml  (2)
└── tests
    ├── test_embeddings_inf2_client.py  (3)
    └── test_embeddings_inf2.py
```
</div>
1. Main python module that defines the [`EmbeddingServiceInf2`](https://github.com/autonomi-ai/nos/blob/main/examples/inf2/embeddings/models/embeddings_inf2.py#L24) model.
2. The [`serve.yaml`](https://github.com/autonomi-ai/nos/blob/main/examples/inf2/embeddings/serve.yaml) serve specification that defines the custom `inf2` runtime and registers the [`EmbeddingServiceInf2`](https://github.com/autonomi-ai/nos/blob/main/examples/inf2/embeddings/models/embeddings_inf2.py#L24) model with NOS.
3. The pytest test for calling the [`EmbeddingServiceInf2`](https://github.com/autonomi-ai/nos/blob/main/examples/inf2/embeddings/models/embeddings_inf2.py#L24) service via gRPC.


The embeddings interface is defined in the [`EmbeddingServiceInf2`](https://github.com/autonomi-ai/nos/blob/main/examples/inf2/embeddings/models/embeddings_inf2.py#L24) module in [`models/embeddings_inf2.py`](https://github.com/autonomi-ai/nos/blob/main/examples/inf2/embeddings/models/embeddings_inf2.py), where the [`__call__`](https://github.com/autonomi-ai/nos/blob/main/examples/inf2/embeddings/models/embeddings_inf2.py#L67) method returns the embedding of the text prompt using [`BAAI/bge-small-en-v1.5`](https://huggingface.co/BAAI/bge-small-en-v1.5) embedding model. 

### 2. Define the custom `inf2` runtime with the NOS serve specification

The [`serve.yaml`](https://github.com/autonomi-ai/nos/blob/main/examples/inf2/embeddings/serve.yaml) serve specification defines the custom embedding model, and a custom `inf2` runtime that NOS uses to execute our model. Follow the annotations below to understand the different components of the serve specification.

<div class="annotate" markdown>

```yaml title="serve.yaml"
images:
  embeddings-inf2:
    base: autonomi/nos:latest-inf2  (1)
    env:
      NOS_LOGGING_LEVEL: DEBUG
      NOS_NEURON_CORES: 2
    run:
      - python -m pip config set global.extra-index-url https://pip.repos.neuron.amazonaws.com
      - pip install sentence-transformers  (2)

models:
  BAAI/bge-small-en-v1.5:
    model_cls: EmbeddingServiceInf2
    model_path: models/embeddings_inf2.py
    default_method: __call__
    runtime_env: embeddings-inf2  (3)
```
</div>
1. Specifies the base runtime image to use - we use the pre-baked `autonomi/nos:latest-inf2` runtime image to build our custom runtime image. This custom NOS runtime comes pre-installed with the AWS Neuron drivers and the AWS Neuron SDK.
2. Installs the `sentence-transformers` library, which is used to embed the text prompt using the `BAAI/bge-small-en-v1.5` model.
3. Specifies the custom runtime environment to use for the specific model deployment - `embeddings-inf2` - which is used to execute the `EmbeddingServiceInf2` model.

In this example, we'll be using the [Huggingface Optimum](https://github.com/huggingface/optimum-neuron) library to help us simplify the deployment process to the Inf2 chip. However, for custom model architectures and optimizations, we have built our own PyTorch tracer and compiler for a growing list of popular models on the [Huggingface Hub](https://huggingface.co/models). 

???question "Need support for custom models on AWS Inferentia2?"
    If you're interested in deploying a custom model on the AWS Inferentia2 chip with NOS, please reach out to us on our [GitHub Issues](https://github.com/autonomi-ai/nos/issues) page or at [support@autonomi.ai](mailto:support@autonomi.ai), and we'll be happy to help you out.

### 3. Deploy the embedding service on AWS `inf2.xlarge` with SkyPilot

Now that we have defined our custom model, let's deploy this service on [AWS Inferentia2](https://aws.amazon.com/en/ec2/instance-types/inf2/) using [SkyPilot](https://github.com/skypilot-org/skypilot). In this example, we're going to use SkyPilot's `sky launch` command to deploy our NOS service on an AWS `inf2.xlarge` on-demand instance. 

Before we launch the service, let's look at the [`job-inf2-embeddings-deployment.yaml`](https://github.com/autonomi-ai/nos/blob/main/examples/inf2/embeddings/job-inf2-embeddings-deployment.yaml) file that we will use to provision the `inf2` instance and deploy the `EmbeddingServiceInf2` model.

<div class="annotate" markdown>

```yaml title="job-inf2-embeddings-deployment.yaml"

file_mounts: (1)
  /app: .

resources:
  cloud: aws
  region: us-west-2
  instance_type: inf2.xlarge (2)
  image_id: ami-096319086cc3d5f23 # us-west-2 (3)
  disk_size: 256
  ports: 8000

setup: |
  sudo apt-get install -y docker-compose-plugin

  cd /app
  cd /app && python3 -m venv .venv && source .venv/bin/activate
  pip install git+https://github.com/autonomi-ai/nos.git pytest (4)

run: |
  source /app/.venv/bin/activate
  cd /app && nos serve up -c serve.yaml --http (5)

```
</div>
1. Mounts the local `./app` directory so that the `serve.yaml`, `models/` and `tests/` directories are available on the remote instance.
2. Specifies the AWS Inferentia2 instance type to use - we use the `inf2.xlarge` instance type.
3. Specifies the Amazon Machine Instance (AMI) use that come pre-installed with AWS Neuron drivers.
4. We simply need `pytest` for testing the client-side logic in `tests/test_embeddings_inf2_client.py`
5. Starts the NOS server with the `serve.yaml` specification. The runtime flag `--runtime inf2` is optional, and automatically detected by NOS as illustrated here.

!!!note "Provisioning `inf2.xlarge` instances"
    To provision an `inf2.xlarge` instance, you will need to have an AWS account and the necessary [service quotas](https://console.aws.amazon.com/servicequotas/home/services/ec2/quotas/) set for the `inf2` instance nodes. For more information on service quotas, please refer to the [AWS documentation](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-resource-limits.html).

!!!warning "Using SkyPilot with `inf2` instances"
    Due to a [job submission bug](https://github.com/skypilot-org/skypilot/issues/2968) in the SkyPilot CLI for `inf2` instances, you will need to use the `skypilot-nightly[aws]` (`pip install skypilot-nightly[aws]`) package to provision `inf2` instances correctly with the `sky launch` command below. 

Let's deploy the `inf2` embeddings service using the following command:
```bash
sky launch -c inf2-embeddings-service job-inf2-embeddings-deployment.yaml
```

???success "`sky launch` output"
    You should see the following output from the `sky launch` command:
    ```bash
    (nos-infra-py38) inf2/embeddings spillai-desktop [ sky launch -c inf2-embeddings-service job-inf2-embeddings-deployment.yaml
    Task from YAML spec: job-inf2-embeddings-deployment.yaml
    I 01-31 21:48:06 optimizer.py:694] == Optimizer ==
    I 01-31 21:48:06 optimizer.py:717] Estimated cost: $0.8 / hour
    I 01-31 21:48:06 optimizer.py:717]
    I 01-31 21:48:06 optimizer.py:840] Considered resources (1 node):
    I 01-31 21:48:06 optimizer.py:910] ------------------------------------------------------------------------------------------
    I 01-31 21:48:06 optimizer.py:910]  CLOUD   INSTANCE      vCPUs   Mem(GB)   ACCELERATORS   REGION/ZONE   COST ($)   CHOSEN
    I 01-31 21:48:06 optimizer.py:910] ------------------------------------------------------------------------------------------
    I 01-31 21:48:06 optimizer.py:910]  AWS     inf2.xlarge   4       16        Inferentia:1   us-west-2     0.76          ✔
    I 01-31 21:48:06 optimizer.py:910] ------------------------------------------------------------------------------------------
    I 01-31 21:48:06 optimizer.py:910]
    Launching a new cluster 'inf2-embeddings-service'. Proceed? [Y/n]: y
    I 01-31 21:48:18 cloud_vm_ray_backend.py:4389] Creating a new cluster: 'inf2-embeddings-service' [1x AWS(inf2.xlarge, {'Inferentia': 1}, image_id={'us-west-2': 'ami-096319086cc3d5f23'}, ports=['8000'])].
    I 01-31 21:48:18 cloud_vm_ray_backend.py:4389] Tip: to reuse an existing cluster, specify --cluster (-c). Run `sky status` to see existing clusters.
    I 01-31 21:48:18 cloud_vm_ray_backend.py:1386] To view detailed progress: tail -n100 -f /home/spillai/sky_logs/sky-2024-01-31-21-48-06-108390/provision.log
    I 01-31 21:48:19 provisioner.py:79] Launching on AWS us-west-2 (us-west-2a,us-west-2b,us-west-2c,us-west-2d)
    I 01-31 21:49:37 provisioner.py:429] Successfully provisioned or found existing instance.
    I 01-31 21:51:03 provisioner.py:531] Successfully provisioned cluster: inf2-embeddings-service
    I 01-31 21:51:04 cloud_vm_ray_backend.py:4418] Processing file mounts.
    I 01-31 21:51:05 cloud_vm_ray_backend.py:4450] To view detailed progress: tail -n100 -f ~/sky_logs/sky-2024-01-31-21-48-06-108390/file_mounts.log
    I 01-31 21:51:05 backend_utils.py:1286] Syncing (to 1 node): . -> ~/.sky/file_mounts/app
    I 01-31 21:51:06 cloud_vm_ray_backend.py:3158] Running setup on 1 node.
    ...
    (task, pid=23904) ✓ Launching docker compose with command: docker compose -f
    (task, pid=23904) /home/ubuntu/.nos/tmp/serve/docker-compose.app.yml up
    (task, pid=23904)  Network serve_default  Creating
    (task, pid=23904)  Network serve_default  Created
    (task, pid=23904)  Container serve-nos-server-1  Creating
    (task, pid=23904)  Container serve-nos-server-1  Created
    (task, pid=23904)  Container serve-nos-http-gateway-1  Creating
    (task, pid=23904)  Container serve-nos-http-gateway-1  Created
    (task, pid=23904) Attaching to serve-nos-http-gateway-1, serve-nos-server-1
    (task, pid=23904) serve-nos-http-gateway-1  | WARNING:  Current configuration will not reload as not all conditions are met, please refer to documentation.
    (task, pid=23904) serve-nos-server-1        |  ✓ InferenceExecutor :: Backend initializing (as daemon) ...
    (task, pid=23904) serve-nos-server-1        |  ✓ InferenceExecutor :: Backend initialized (elapsed=2.9s).
    (task, pid=23904) serve-nos-server-1        |  ✓ InferenceExecutor :: Connected to backend.
    (task, pid=23904) serve-nos-server-1        |  ✓ Starting gRPC server on [::]:50051
    (task, pid=23904) serve-nos-server-1        |  ✓ InferenceService :: Deployment complete (elapsed=0.0s)
    (task, pid=23904) serve-nos-server-1        | (EmbeddingServiceInf2 pid=404) 2024-01-31 21:53:58.566 | INFO     | nos.neuron.device:setup_environment:36 - Setting up neuron env with 2 cores
    ...
    (task, pid=23904) serve-nos-server-1        | (EmbeddingServiceInf2 pid=404) 2024-02-01T05:54:36Z Compiler status PASS
    (task, pid=23904) serve-nos-server-1        | (EmbeddingServiceInf2 pid=404) 2024-01-31 21:54:46.928 | INFO     | EmbeddingServiceInf2:__init__:61 - Saved model to /app/.nos/cache/neuron/BAAI/bge-small-en-v1.5-bs-1-sl-384
    (task, pid=23904) serve-nos-server-1        | (EmbeddingServiceInf2 pid=404) 2024-01-31 21:54:47.037 | INFO     | EmbeddingServiceInf2:__init__:64 - Loaded neuron model: BAAI/bge-small-en-v1.5
    ...
    (task, pid=23904) serve-nos-server-1        | 2024-01-31 22:25:43.710 | INFO     | nos.server._service:Run:360 - Executing request [model=BAAI/bge-small-en-v1.5, method=None]
    (task, pid=23904) serve-nos-server-1        | 2024-01-31 22:25:43.717 | INFO     | nos.server._service:Run:362 - Executed request [model=BAAI/bge-small-en-v1.5, method=None, elapsed=7.1ms]
    ```

Once complete, you should see the following (trimmed) output from the `sky launch` command:
```bash
✓ InferenceExecutor :: Backend initializing (as daemon) ...
✓ InferenceExecutor :: Backend initialized (elapsed=2.9s).
✓ InferenceExecutor :: Connected to backend.
✓ Starting gRPC server on [::]:50051
✓ InferenceService :: Deployment complete (elapsed=0.0s)
Setting up neuron env with 2 cores
...
Compiler status PASS
Saved model to /app/.nos/cache/neuron/BAAI/bge-small-en-v1.5-bs-1-sl-384
Loaded neuron model: BAAI/bge-small-en-v1.5
```


### 3. Test your custom model on AWS Inf2 instance

Once the service is deployed, you should be able to simply make a cURL request to the `inf2` instance to test the server-side logic of the embeddings model.

=== "Using cURL (remote)"

    ```bash
    export IP=$(sky status --ip inf2-embeddings-service)

    curl \
    -X POST http://${IP}:8000/v1/infer \
    -H 'Content-Type: application/json' \
    -d '{
        "model_id": "BAAI/bge-small-en-v1.5",
        "inputs": {
            "texts": ["fox jumped over the moon"]
        }
    }'
    ```

=== "Using the gRPC client (on the `inf2` instance)"

    Optionally, you can also test the gRPC service using the provided [`tests/test_embeddings_inf2_client.py`](https://github.com/autonomi-ai/nos/blob/main/examples/inf2/embeddings/tests/test_embeddings_inf2_client.py). For this test however, you'll need to ssh into the `inf2` instance and run the following command.

    ```bash
    ssh inf2-embeddings-service
    ```

    Once you're on the `inf2.xlarge` instance, you can run `pytest -sv tests/test_embeddings_inf2_client.py` to test the server-side logic of the embeddings model. 
    
    ```bash
    $ pytest -sv tests/test_embeddings_inf2_client.py
    ```
    
    Here's a simplified version of the test to execute the embeddings model.

    ```python
    from nos.client import Client

    # Create the client
    client = Client("[::]:50051")
    assert client.WaitForServer()

    # Load the embeddings model
    model = client.Module("BAAI/bge-small-en-v1.5")

    # Embed text with the model
    texts = "What is the meaning of life?"
    response = model(texts=texts)
    ```


## 🤑 What's it going to cost me?

The table below shows the costs of deploying one of these *latency-optimized* (`bsize=1`) embedding services on a single Inf2 instance on AWS. While the costs are only one part of the equation, it is important to note that the AWS Inf2 instances are ~25% cheaper than the NVIDIA A10G instances, and offer a more cost-effective solution for inference workloads on AWS. In the coming weeks, we'll be digging into evaluating the performance of the Inf2 instances with respect to their NVIDIA GPU counterparts on inference metrics such as latency/throughput and cost metrics such as number of requests / $, montly costs etc.

| Model | Cloud Instance | Spot | Cost / hr | Cost / month | # of Req. / $ | 
| ----- | -------------- | ---- | --------- | ------------ | ---------- |
| [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5) | `inf2.xlarge` | - | $0.75 | ~$540 | ~685K / $1 | 
| **[BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5)** | `inf2.xlarge` | ✅ | **$0.32** | **~$230** | ~1.6M / $1 | 


## 🎁 Wrapping up

In this post, we introduced the new NOS **[`inf2`](../../concepts/runtime-environments.md#🏃‍♂️-supported-runtimes)** runtime that allows developers to easily develop, and serve models on the [AWS Inferentia2](https://aws.amazon.com/en/ec2/instance-types/inf2/) chip. With more cost-efficient, and inference-optimized chips coming to market ([Google TPUs](https://cloud.google.com/tpu/docs/v5e-inference), [Groq](https://groq.com/products/), [Tenstorrent](https://tenstorrent.com/cards/) etc), we believe it is important for developers to be able to easily access and deploy models on these devices. The specialized [NOS Inference Runtime](../../concepts/runtime-environments.md#⚡️-nos-inference-runtime) aims to do just that - a fast, and frictionless way to deploy models onto any of the AI accelerators, be it NVIDIA GPUs or AWS Inferentia2 chips, in the cloud, or on-prem.

Thanks for reading, and we hope you found this post useful - and finally, give [NOS](https://github.com/autonomi-ai/nos) a try. If you have any questions, or would like to learn more about the [NOS](https://github.com/autonomi-ai/nos) `inf2` runtime, please reach out to us on our [GitHub Issues](https://github.com/autonomi-ai/nos/issues) page or join us on [Discord](https://discord.gg/QAGgvTuvgg). 
<br>