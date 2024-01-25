# Serving with `docker` and `docker-compose`

This tutorial shows how to serve the NOS server directly with `docker` or `docker-compose`.

## Serving with `docker`

To run the NOS gRPC server with `docker` simply run:

For CPU:
```sh
docker run --rm \
    -e NOS_HOME=/app/.nos \
    -v $(HOME)/.nos:/app/.nos \
    -v /dev/shm:/dev/shm \
    -p 50051:50051 \
    autonomi/nos:latest-cpu
```

For running the GPU server, you need to install `nvidia-docker` and run the following command:
```sh
docker run --rm \
    --name nos-grpc-server \
    --gpus all \
    -e NOS_HOME=/app/.nos \
    -v $(HOME)/.nos:/app/.nos \
    -v /dev/shm:/dev/shm \
    -p 50051:50051 \
    autonomi/nos:latest-gpu
```


## Serving with `docker-compose`

To run the NOS gRPC server and the HTTP gateway with `docker-compose` simply run:

```sh
docker-compose up -f docker-compose.yml
```

You should now see the logs both from the server and the gateway.

```bash
(nos-py38) tutorials/05-serving-with-docker desktop [ docker compose -f docker-compose.yml up
[+] Running 2/2
 ✔ Container 05-serving-with-docker-nos-grpc-server-1   Created                                                                                                                                                0.0s
 ✔ Container 05-serving-with-docker-nos-http-gateway-1  Recreated                                                                                                                                              0.0s
Attaching to 05-serving-with-docker-nos-grpc-server-1, 05-serving-with-docker-nos-http-gateway-1
05-serving-with-docker-nos-grpc-server-1   | Starting server with OMP_NUM_THREADS=64...
05-serving-with-docker-nos-http-gateway-1  | WARNING:  Current configuration will not reload as not all conditions are met, please refer to documentation.
05-serving-with-docker-nos-grpc-server-1   |  ✓ InferenceExecutor :: Connected to backend.
05-serving-with-docker-nos-grpc-server-1   |  ✓ Starting gRPC server on [::]:50051
05-serving-with-docker-nos-grpc-server-1   |  ✓ InferenceService :: Deployment complete (elapsed=0.0s)
05-serving-with-docker-nos-http-gateway-1  | INFO:     Started server process [1]
05-serving-with-docker-nos-http-gateway-1  | INFO:     Waiting for application startup.
05-serving-with-docker-nos-http-gateway-1  | INFO:     Application startup complete.
05-serving-with-docker-nos-http-gateway-1  | INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

The HTTP gateway service (`nos-http-gateway`) in the `docker-compose.yml` file simply forwards the HTTP requests to the gRPC server (`nos-grpc-server`). This is especially useful when exposing the server via a REST API.

Here's the full `docker-compose.yml` file:

```yaml
{% include './docker-compose.yml' %}
```

## Testing the server

To test the server's health, you can simply use `curl`:

```sh
curl -X "GET" "http://localhost:8000/v1/health" -H "accept: application/json"
```

You should see the following response:
```sh
{"status":"ok"}
```

You can now try one of the many requests showcased in the main [README](../../../README.md#what-can-nos-do).

## Debugging the server

 - **Running on CPUs:** You can remove the `deploy` section from the `docker-compose.yml` file to run the server without GPU capabilities.
 - **Running on GPUs:** Make sure you have `nvidia-docker` installed and that you have the latest NVIDIA drivers installed on your machine. You can check the NVIDIA drivers version by running `nvidia-smi` on your terminal. If you don't have `nvidia-docker` installed, you can follow the instructions [here](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker).
 - **Running on MacOS:** You can run the server on MacOS by removing the `deploy` section from the `docker-compose.yml` file.
 - **Enabling debug logs:** You can enable debug logs on both the docker services by setting the `NOS_LOGGING_LEVEL` environment variable to `DEBUG` in the `docker-compose.yml` file. This should provide you with more information on what's happening under the hood.
