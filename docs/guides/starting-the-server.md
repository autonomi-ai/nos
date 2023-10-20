The NOS gRPC server can be started in two ways:

- Via the NOS SDK using [`nos.init(...)`](#api-reference) **(preferred for development)**
- Via Docker Compose **(recommended for production deployments)**

You can also start the server with the REST API proxy enabled as shown in the 3rd example below.

=== "Via SDK"

    You can start the nos server programmatically via the NOS SDK:
    ```python
    import nos

    nos.init(runtime="auto")
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
    Navigate to [`examples/skypilot`](https://github.com/autonomi-ai/nos/nos/examples/skypilot) to see an example of the YAML specification. You can start the server with the following command:

    ```bash
    docker-compose -f docker-compose.gpu.yml up
    ```

    ```yaml title="docker-compose.gpu.yml"
    {% include '../../examples/docker/docker-compose.gpu.yml' %}
    ```


## API Reference
---
::: nos.init
::: nos.shutdown
