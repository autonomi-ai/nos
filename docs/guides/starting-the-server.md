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
    Navigate to [`examples/quickstart`](https://github.com/autonomi-ai/nos/nos/examples/quickstart) to see an example of the YAML specification.

    ```yaml
    {% include '../../examples/quickstart/docker-compose.quickstart.yml' %}
    ```

## API Reference
---
::: nos.init
::: nos.shutdown
