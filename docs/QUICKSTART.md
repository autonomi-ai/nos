# 🔥 Quickstart

1. **Install NOS**

    ```bash
    pip install autonomi-nos[torch]
    ```

    Alternatively, if you have `torch` already installed, you can simply run:
    ```bash
    pip install autonomi-nos
    ```

2. **Run the NOS server with Docker**

    Navigate to the `examples/quickstart` folder and run the NOS server via:
    ```bash
    docker compose -f docker-compose.quickstart.yml up
    ```

    Let's inspect the docker-compose file to understand what's going on:
    ```yaml
    {% include "../examples/quickstart/docker-compose.quickstart.yml" %}
    ```

    We first spin up a `nos-server` service mounting the necessary host directories (`~/.nosd`) and exposing the gRPC port. The command `nos-grpc-server` spins up the gRPC server with the default 50051 port that can be used to send inference requests. The `NOS_HOME` directory is set to `/app/.nos` where all the models and optimization artifacts are stored. This directory is mounted on your host machine at `~/.nosd`. In addition, we also create a `tmp` volume for mounting the container's `/tmp` directory.

    Alternatively, you can also get started with docker via:
    ```bash
    docker run -it \
        -p 50051:50051 \
        -v ~/.nosd:/app/.nos \
        --shm-size 4g \
        autonomi/nos:latest-cpu
    ```

3. **Run Inference**
    Try out an inference request via the CLI or [Python SDK](https://pypi.org/project/autonomi-nos):

    **Via CLI**
    ```bash
    nos serve-grpc txt2img -i "dog riding horse"
    ```

    **Via [Python SDK](https://pypi.org/project/autonomi-nos)**
    ```python
    from nos.client import InferenceClient

    client = InferenceClient("localhost:50051")
    response = client.Predict(
        method="txt2img",
        model="stabilityai/stable-diffusion-2",
        text="dog riding horse")
    img = response["image"]
    ```
