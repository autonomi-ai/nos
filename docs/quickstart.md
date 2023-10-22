# 🔥 Quickstart

## 🛠️ Install Dependencies

You will need to install [Docker](https://docs.docker.com/get-docker/), [Nvidia Docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker) and [Docker Compose](https://docs.docker.com/compose/install/).

=== "Linux (Debian/Ubuntu)"
    On Linux, you can install Docker and Docker Compose via the following commands:
    ```bash
    sudo apt-get update \
    sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin \
    && sudo systemctl restart docker
    ```

    Next, let's install Nvidia Docker. This will install the Nvidia Container Toolkit which is required to run GPU accelerated containers. This is only required if you plan to run the NOS server with GPU support.
    ```bash
    sudo apt-get update nvidia-container-toolkit-base
    ```

    Finally, you should be able to run the following command without any errors and the `nvidia-smi` output:
    ```bash
    docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
    ```

    If you run into issues, refer to the official Nvidia install [guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) or just ping us on [#nos-support](https://discord.gg/qEvfUcgS5m).

=== "MacOS (Intel/Apple Silicon)"
    Download and install [Docker Desktop](https://docs.docker.com/desktop/mac/install/) directly from the Docker website.

## 👩‍💻 Install NOS

We highly recommend doing all of the following inside of a Conda or Virtualenv environment. You can install Conda on your machine following the official [guide](https://conda.io/projects/conda/en/latest/user-guide/install/index.html). Create a new env:
```bash
conda create -n nos python=3.8
conda activate nos
```

=== "Client-only Installation"

    ``` sh
    pip install torch-nos
    ```

=== "Server Installation (GPU)"

    If you plan to run the NOS server locally (i.e. outside docker), you will also need to install the `server` extra dependencies:
    ```sh
    pip install torch-nos[server]"
    ```

    !!!note
        We currently only support running the NOS server on Linux with GPUs. 

    !!!note
        Python 3.8 is currently required to run the server on MacOS due to Ray requirements. If you don't plan to run the server locally then this requirement can be relaxed.

## ⚡️ Start the NOS backend server

You can start the nos server programmatically via the NOS SDK:

```python
import nos

nos.init(runtime="auto")
```

We're now ready to issue our first inference request with NOS!

## 🚀 Run Inference

Try out an inference request via the [Python SDK](https://pypi.org/project/torch-nos):

```python
from nos.client import Client, TaskType

client = Client()
response = client.Run(
    "stabilityai/stable-diffusion-2",
    inputs={
        "texts"=["astronaut on the moon"],
        "num_images": 1, "width": 512, "height": 512
    })
img, = response["images"]
```

If you run into issues after following this guide, feel free to ping us on [#nos-support](https://discord.gg/qEvfUcgS5m).
