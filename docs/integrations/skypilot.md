In this guide we'll show you how you can deploy the NOS inference server using [SkyPilot](https://skypilot.readthedocs.io/) on any of the popular Cloud Service Providers (CSPs) such as AWS, GCP or Azure. We'll use GCP as an example, but the steps are similar for other CSPs.

!!!quote "What is SkyPilot?"
    [SkyPilot](https://skypilot.readthedocs.io/) is a framework for running LLMs, AI, and batch jobs on any cloud, offering maximum cost savings, highest GPU availability, and managed execution. - [SkyPilot Documentation](https://skypilot.readthedocs.io/).

## 👩‍💻 Prerequisites

You'll first need to install [SkyPilot](https://skypilot.readthedocs.io/) in your virtual environment / conda environment before getting started. Before getting started, we recommend you go through their [quickstart](https://skypilot.readthedocs.io/en/latest/getting-started/quickstart.html) to familiarize yourself with the SkyPilot tool.

```bash
$ pip install skypilot[gcp]
```

If you're installing SkyPilot for use with other cloud providers, you may install any of the relevant extras `skypilot[aws,gcp,azure]`. `

### [OPTIONAL] Configure cloud credentials

Run `sky check` for more details and installation instructions.

## 📦 Deploying NOS on GCP

### 1. Define your SkyPilot deployment YAML

First, let's create a `sky.yaml` YAML file with the following configuration. 

```yaml
{% include '../../examples/skypilot/sky.yaml' %}
```

Here, we are going to provision a single GPU server on GCP with an NVIDIA T4 GPU and expose ports `8000` (REST) and `50051` (gRPC) for the NOS server. 

### 🚀 2. Launch your NOS server

Now, we can launch our NOS server on GCP with the following command:

```bash
$ sky launch -c nos-server sky.yaml
```

That's it! You should see the following output:

```bash
(nos-infra-py38) examples/skypilot spillai-desktop [ sky launch -c nos-server sky.yaml --cloud gcp
Task from YAML spec: sky.yaml
I 01-16 09:41:18 optimizer.py:694] == Optimizer ==
I 01-16 09:41:18 optimizer.py:705] Target: minimizing cost
I 01-16 09:41:18 optimizer.py:717] Estimated cost: $0.6 / hour
I 01-16 09:41:18 optimizer.py:717]
I 01-16 09:41:18 optimizer.py:840] Considered resources (1 node):
I 01-16 09:41:18 optimizer.py:910] ---------------------------------------------------------------------------------------------
I 01-16 09:41:18 optimizer.py:910]  CLOUD   INSTANCE       vCPUs   Mem(GB)   ACCELERATORS   REGION/ZONE     COST ($)   CHOSEN
I 01-16 09:41:18 optimizer.py:910] ---------------------------------------------------------------------------------------------
I 01-16 09:41:18 optimizer.py:910]  GCP     n1-highmem-4   4       26        T4:1           us-central1-a   0.59          ✔
I 01-16 09:41:18 optimizer.py:910] ---------------------------------------------------------------------------------------------
I 01-16 09:41:18 optimizer.py:910]
Launching a new cluster 'nos-server'. Proceed? [Y/n]: y
I 01-16 09:41:25 cloud_vm_ray_backend.py:4508] Creating a new cluster: 'nos-server' [1x GCP(n1-highmem-4, {'T4': 1}, ports=['8000', '50051'])].
I 01-16 09:41:25 cloud_vm_ray_backend.py:4508] Tip: to reuse an existing cluster, specify --cluster (-c). Run `sky status` to see existing clusters.
I 01-16 09:41:26 cloud_vm_ray_backend.py:1474] To view detailed progress: tail -n100 -f /home/spillai/sky_logs/sky-2024-01-16-09-41-16-157615/provision.log
I 01-16 09:41:29 cloud_vm_ray_backend.py:1912] Launching on GCP us-central1 (us-central1-a)
I 01-16 09:44:36 log_utils.py:45] Head node is up.
I 01-16 09:45:43 cloud_vm_ray_backend.py:1717] Successfully provisioned or found existing VM.
I 01-16 09:46:00 cloud_vm_ray_backend.py:4558] Processing file mounts.
I 01-16 09:46:00 cloud_vm_ray_backend.py:4590] To view detailed progress: tail -n100 -f ~/sky_logs/sky-2024-01-16-09-41-16-157615/file_mounts.log
I 01-16 09:46:00 backend_utils.py:1459] Syncing (to 1 node): ./app -> ~/.sky/file_mounts/app
I 01-16 09:46:05 cloud_vm_ray_backend.py:3315] Running setup on 1 node.
...
...
...
(nos-server, pid=12112) Status: Downloaded newer image for autonomi/nos:0.1.4-gpu
(nos-server, pid=12112) docker.io/autonomi/nos:0.1.4-gpu
(nos-server, pid=12112) 2024-01-16 17:49:09.415 | INFO     | nos.server:_pull_image:235 - Pulled new server image: autonomi/nos:0.1.4-gpu
(nos-server, pid=12112) ✓ Successfully generated docker-compose file
(nos-server, pid=12112) (filename=docker-compose.sky_workdir.yml).
(nos-server, pid=12112) ✓ Launching docker compose with command: docker compose -f
(nos-server, pid=12112) /home/gcpuser/.nos/tmp/serve/docker-compose.sky_workdir.yml up
(nos-server, pid=12112)  Container serve-nos-server-1  Creating
(nos-server, pid=12112)  Container serve-nos-server-1  Created
(nos-server, pid=12112)  Container serve-nos-http-gateway-1  Creating
(nos-server, pid=12112)  Container serve-nos-http-gateway-1  Created
(nos-server, pid=12112) Attaching to serve-nos-http-gateway-1, serve-nos-server-1
(nos-server, pid=12112) serve-nos-server-1        | Starting server with OMP_NUM_THREADS=4...
(nos-server, pid=12112) serve-nos-http-gateway-1  | WARNING:  Current configuration will not reload as not all conditions are met, please refer to documentation.
(nos-server, pid=12112) serve-nos-server-1        |  ✓ InferenceExecutor :: Connected to backend.
(nos-server, pid=12112) serve-nos-server-1        |  ✓ Starting gRPC server on [::]:50051
(nos-server, pid=12112) serve-nos-server-1        |  ✓ InferenceService :: Deployment complete (elapsed=0.0s)
(nos-server, pid=12112) serve-nos-http-gateway-1  | INFO:     Started server process [1]
(nos-server, pid=12112) serve-nos-http-gateway-1  | INFO:     Waiting for application startup.
(nos-server, pid=12112) serve-nos-http-gateway-1  | INFO:     Application startup complete.
(nos-server, pid=12112) serve-nos-http-gateway-1  | INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

### 🔋 3. Check the status of your NOS server

You can check the status of your NOS server with the following command:

```bash
$ sky status
```

You should see the following output:

```bash
NAME                            LAUNCHED     RESOURCES                                                                  STATUS   AUTOSTOP  COMMAND
nos-server                      1 min ago    1x GCP(n1-highmem-4, {'T4': 1}, ports=[8000, 50051])                       UP       -         sky launch -c nos-server-...
```

Congratulations! You've successfully deployed your NOS server on GCP. You can now access the NOS server from your local machine at `<ip>:8000` or `<ip>:50051`. In a new terminal, let's check the health of our NOS server with the following command:

```bash
$ curl http://$(sky status --ip nos-server):8000/v1/health
```

You should see the following output:

```bash
{"status": "ok"}
```

### 💬 4. Chat with your hosted LLM endpoint

You can now chat with your hosted LLM endpoint using the following command:

```bash
curl \
-X POST http://$(sky status --ip nos-server):8000/v1/chat/completions \
-H "Content-Type: application/json" \
-d '{
    "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "messages": [{"role": "user", "content": "Tell me a story of 1000 words with emojis"}],
    "temperature": 0.7, "stream": true
  }'
```

On the first call to the server, the server will download the model from Huggingface, cache it locally and load it onto the GPU. Subsequent calls will not have any of this overhead as the GPU memory for the models will be pinned.


### 🛑 5. Stop / Terminate your NOS server

Once you're done using your server, you can stop it with the following command:

```bash
$ sky stop nos-server-gcp
```

Alternatively, you can terminate your server with the following command:

```bash
$ sky down nos-server-gcp
```

This will terminate the server and all associated resources (e.g. VMs, disks, etc.).
