# 🔥 Quickstart

0. **Dependencies**

    We highly reccomend doing all of the following inside of a Conda environment. Install Conda on your machine following the official [guide](https://conda.io/projects/conda/en/latest/user-guide/install/index.html). Create a new env:
    ```bash
    conda create -n nos python=3.8
    ```

    Python 3.8 is currently required to run the server on MacOS due to Ray support.

    Install Pip as well if its missing:
    ```bash
    conda install pip
    ```

1. **Install NOS**

    ```bash
    pip install autonomi-nos[torch]
    ```

    Alternatively, if you have `torch` already installed, you can simply run:
    ```bash
    pip install autonomi-nos
    ```

2. **(OPTIONAL) Install Docker dependencies for local NOS server**

    If you are running the NOS container locally on a linux box, you will also need to install Docker
    and Nvidia Docker.
    ```bash
    sudo apt-get update \
    && sudo apt-get install -y nvidia-container-toolkit-base
    ```
    If you run into issues, refer to the official Nvidia install [guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) or just ping us here.

2. **Run the NOS server with Docker**

    Start the nos server with the appropriate backend:
    ```bash
    nos docker start --runtime=[gpu, cpu]
    ```

    This will spin up `nos-nos-server` in `docker ps`. We're now ready to issue
    out first inference request!

3. **Run Inference**
    Try out an inference request via the CLI or [Python SDK](https://pypi.org/project/autonomi-nos):

    **Via CLI**
    ```bash
    nos predict txt2img -i "dog riding horse"
    ```

    **Via [Python SDK](https://pypi.org/project/autonomi-nos)**
    ```python
    from nos.client import InferenceClient, TaskType

    client = InferenceClient()
    response = client.Run(
        task=TaskType.IMAGE_GENERATION
        model="stabilityai/stable-diffusion-2",
        texts=["dog riding horse"])
    img = response["image"]
    ```
