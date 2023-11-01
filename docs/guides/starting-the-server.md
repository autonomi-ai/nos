The NOS gRPC server can be started in two ways:

- Via the NOS SDK using [`nos.init(...)`](#api-reference) **(preferred for development)**
- Via the NOS `serve` CLI.
- Via Docker Compose **(recommended for production deployments)**

You can also start the server with the REST API proxy enabled as shown in the 2nd and 4th examples below.

=== "Via SDK"

    You can start the nos server programmatically via the NOS SDK:
    ```python
    import nos

    nos.init(runtime="auto")
    ```
=== "Via CLI"

    You can start the nos server via the NOS `serve` CLI:
    ```bash
    nos serve up
    ```

    Optionally, to use the REST API, you can start an HTTP gateway proxy alongside the gRPC server:
    ```bash
    nos serve up --http
    ```

=== "Via Docker Compose (gRPC)"
    Navigate to [`examples/docker`](https://github.com/autonomi-ai/nos/nos/examples/docker) to see an example of the YAML specification. You can start the server with the following command:

    ```bash
    docker-compose -f docker-compose.gpu.yml up
    ```

    ```yaml title="docker-compose.gpu.yml"
    {% include '../../examples/docker/docker-compose.gpu.yml' %}
    ```

=== "Via Docker Compose (gRPC + REST)"
    Navigate to [`examples/docker`](https://github.com/autonomi-ai/nos/nos/examples/docker) to see an example of the YAML specification. You can start the server with the following command:

    ```bash
    docker-compose -f docker-compose.gpu-with-gateway.yml up
    ```

    ```yaml title="docker-compose.gpu-with-gateway.yml"
    {% include '../../examples/docker/docker-compose.gpu-with-gateway.yml' %}
    ```


## API Reference
---
::: nos.init
::: nos.shutdown
