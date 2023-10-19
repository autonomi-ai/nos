The NOS server can be started in two ways:

- Via the NOS SDK using `nos.init(...)` **(preferred for development)**
- Via Docker Compose **(recommended for production deployments)**

=== "Via SDK"

    You can start the nos server programmatically via the NOS SDK:
    ```python
    import nos

    nos.init(runtime="auto")
    ```

=== "Via Docker Compose"
    Navigate to [`examples/skypilot`](https://github.com/autonomi-ai/nos/nos/examples/skypilot) to see an example of the YAML specification.

    ```yaml
    {% include '../../examples/skypilot/app/docker-compose.gpu.yml' %}
    ```

## API Reference
---
::: nos.init
::: nos.shutdown
