In this guide we'll show you how you can deploy the NOS inference server using [SkyPilot](https://skypilot.readthedocs.io/) on any of the popular Cloud Service Providers (CSPs) such as AWS, GCP or Azure. We'll use GCP as an example, but the steps are similar for other CSPs.

## 👩‍💻 Prerequisites

You'll first need to install [SkyPilot](https://skypilot.readthedocs.io/) in your virtual environment / conda environment before getting started. Before getting started, we recommend you go through their [quickstart](https://skypilot.readthedocs.io/en/latest/getting-started/quickstart.html) to familiarize yourself with SkyPilot.

```bash
$ conda env create -n nos python=3.8
$ conda activate nos
$ pip install skypilot[gcp]
```

If you're installing SkyPilot for use with other cloud providers, you may install any of the relevant extras `skypilot[aws,gcp,azure]`. `

### [OPTIONAL] Configure cloud credentials

Run `sky check` for more details and installation instructions.

## 📦 Deploying NOS on GCP

### 1. Define your SkyPilot deployment YAML

First, let's create a `server.yaml` YAML file with the following configuration. Here, we are going to provision a single GPU server on GCP (`us-west1`) with an NVIDIA T4 GPU and expose ports `8000` and `50051` for the NOS server. The `docker-compose.gpu.yml` also exposes these ports, so we can access the NOS server from our local machine. We also mount the current directory `.` to `/app` in the container so that we can access the `docker-compose.gpu.yml` file from within the container.

```yaml
{% include '../../examples/skypilot/app/docker-compose.gpu.yml' %}
```

### 🚀 2. Launch your NOS server

Now, we can launch our NOS server on GCP with the following command:

```bash
$ sky launch -c nos-server-gcp server.yaml
```

That's it! You should see the following output:

```bash
Task from YAML spec: server.yaml
I 10-18 15:55:05 optimizer.py:652] == Optimizer ==
I 10-18 15:55:05 optimizer.py:663] Target: minimizing cost
I 10-18 15:55:05 optimizer.py:675] Estimated cost: $0.6 / hour
I 10-18 15:55:05 optimizer.py:675]
I 10-18 15:55:05 optimizer.py:748] Considered resources (1 node):
I 10-18 15:55:05 optimizer.py:797] -------------------------------------------------------------------------------------------
I 10-18 15:55:05 optimizer.py:797]  CLOUD   INSTANCE       vCPUs   Mem(GB)   ACCELERATORS   REGION/ZONE   COST ($)   CHOSEN
I 10-18 15:55:05 optimizer.py:797] -------------------------------------------------------------------------------------------
I 10-18 15:55:05 optimizer.py:797]  GCP     n1-highmem-4   4       26        T4:1           us-west1-a    0.59          ✔
I 10-18 15:55:05 optimizer.py:797] -------------------------------------------------------------------------------------------
I 10-18 15:55:05 optimizer.py:797]
Launching a new cluster 'nos-server-gcp'. Proceed? [Y/n]: y
I 10-18 15:56:38 cloud_vm_ray_backend.py:4172] Creating a new cluster: "nos-server-gcp" [1x GCP(n1-highmem-4, {'T4': 1}, ports=[8000, 50051])].
I 10-18 15:56:38 cloud_vm_ray_backend.py:4172] Tip: to reuse an existing cluster, specify --cluster (-c). Run `sky status` to see existing clusters.
I 10-18 15:56:39 cloud_vm_ray_backend.py:1427] To view detailed progress: tail -n100 -f /home/spillai/sky_logs/sky-2023-10-18-15-55-03-683434/provision.log
I 10-18 15:56:42 cloud_vm_ray_backend.py:1807] Launching on GCP us-west1 (us-west1-a)
I 10-18 16:00:11 log_utils.py:89] Head node is up.
I 10-18 16:01:17 cloud_vm_ray_backend.py:1616] Successfully provisioned or found existing VM.
I 10-18 16:01:22 cloud_vm_ray_backend.py:4222] Processing file mounts.
I 10-18 16:01:22 cloud_vm_ray_backend.py:4254] To view detailed progress: tail -n100 -f ~/sky_logs/sky-2023-10-18-15-55-03-683434/file_mounts.log
I 10-18 16:01:22 backend_utils.py:1424] Syncing (to 1 node): . -> ~/.sky/file_mounts/app
I 10-18 16:01:26 cloud_vm_ray_backend.py:3063] Running setup on 1 node.
...
Attaching to app-nos-server-1
app-nos-server-1  | + nproc --all
app-nos-server-1  | + NCORES=4
app-nos-server-1  | + echo Starting Ray server with OMP_NUM_THREADS=4...
app-nos-server-1  | + OMP_NUM_THREADS=4 ray start --head
app-nos-server-1  | Starting Ray server with OMP_NUM_THREADS=4...
app-nos-server-1  | 2023-10-18 16:08:38,064     INFO usage_lib.py:381 -- Usage stats collection is disabled.
app-nos-server-1  | 2023-10-18 16:08:38,065     INFO scripts.py:722 -- Local node IP: 172.18.0.2
app-nos-server-1  | 2023-10-18 16:08:41,967     SUCC scripts.py:759 -- --------------------
app-nos-server-1  | 2023-10-18 16:08:41,968     SUCC scripts.py:760 -- Ray runtime started.
app-nos-server-1  | 2023-10-18 16:08:41,968     SUCC scripts.py:761 -- --------------------                                                                                        app-nos-server-1  | 2023-10-18 16:08:41,969     INFO scripts.py:763 -- Next steps
app-nos-server-1  | 2023-10-18 16:08:41,970     INFO scripts.py:766 -- To add another node to this Ray cluster, run
app-nos-server-1  | 2023-10-18 16:08:41,970     INFO scripts.py:769 --   ray start --address='172.18.0.2:6379'
app-nos-server-1  | 2023-10-18 16:08:41,970     INFO scripts.py:778 -- To connect to this Ray cluster:
app-nos-server-1  | 2023-10-18 16:08:41,971     INFO scripts.py:780 -- import ray
app-nos-server-1  | 2023-10-18 16:08:41,971     INFO scripts.py:781 -- ray.init()
app-nos-server-1  | 2023-10-18 16:08:41,972     INFO scripts.py:793 -- To submit a Ray job using the Ray Jobs CLI:
app-nos-server-1  | 2023-10-18 16:08:41,972     INFO scripts.py:794 --   RAY_ADDRESS='http://127.0.0.1:8265' ray job submit --working-dir . -- python my_script.py
app-nos-server-1  | 2023-10-18 16:08:41,973     INFO scripts.py:803 -- See https://docs.ray.io/en/latest/cluster/running-applications/job-submission/index.html
app-nos-server-1  | 2023-10-18 16:08:41,973     INFO scripts.py:807 -- for more information on submitting Ray jobs to the Ray cluster.
app-nos-server-1  | 2023-10-18 16:08:41,974     INFO scripts.py:812 -- To terminate the Ray runtime, run
app-nos-server-1  | 2023-10-18 16:08:41,974     INFO scripts.py:813 --   ray stop
app-nos-server-1  | 2023-10-18 16:08:41,975     INFO scripts.py:816 -- To view the status of the cluster, use
app-nos-server-1  | 2023-10-18 16:08:41,975     INFO scripts.py:817 --   ray status
app-nos-server-1  | 2023-10-18 16:08:41,976     INFO scripts.py:821 -- To monitor and debug Ray, view the dashboard at
app-nos-server-1  | 2023-10-18 16:08:41,976     INFO scripts.py:822 --   127.0.0.1:8265
app-nos-server-1  | 2023-10-18 16:08:41,976     INFO scripts.py:829 -- If connection to the dashboard fails, check your firewall settings and network configuration.
app-nos-server-1  | + echo Starting NOS server...
app-nos-server-1  | + nos-grpc-server
app-nos-server-1  | Starting NOS server...
app-nos-server-1  | 2023-10-18 16:08:53,364     INFO worker.py:1431 -- Connecting to existing Ray cluster at address: 172.18.0.2:6379...
app-nos-server-1  | 2023-10-18 16:08:53,372     INFO worker.py:1612 -- Connected to Ray cluster. View the dashboard at http://127.0.0.1:8265
app-nos-server-1  |  ✓ InferenceExecutor :: Connected to backend.
app-nos-server-1  |  Starting server on [::]:50051
app-nos-server-1  |  ✓ InferenceService :: Deployment complete (elapsed=0.0s)
```

### 🔋 3. Check the status of your NOS server

You can check the status of your NOS server with the following command:

```bash
$ sky status
```

You should see the following output:

```bash
NAME                            LAUNCHED     RESOURCES                                                                  STATUS   AUTOSTOP  COMMAND
nos-server-gcp                  1 min ago    1x GCP(n1-highmem-4, {'T4': 1}, ports=[8000, 50051])                       UP       -         sky launch -c nos-server-...
```

Congratulations! You've successfully deployed your NOS server on GCP. You can now access the NOS server from your local machine at `<ip>:8000` or `<ip>:50051`.

### 🛑 4. Stop / Terminate your NOS server

Once you're done using your server, you can stop it with the following command:

```bash
$ sky stop nos-server-gcp
```

Alternatively, you can terminate your server with the following command:

```bash
$ sky terminate nos-server-gcp
```

This will terminate the server and all associated resources (e.g. VMs, disks, etc.).
