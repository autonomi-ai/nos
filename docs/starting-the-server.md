The NOS server can be started in three ways:

- Via the NOS SDK **(preferred for development)**
- Via Docker Compose (recommended for production deployments)
- Via CLI (mostly for convenience)

=== "Via SDK"

    You can start the nos server programmatically via the NOS SDK:
    ```python
    import nos

    nos.init(runtime="auto")
    ```

=== "Via CLI"

    Start the nos server with the appropriate backend:
    ```bash
    nos docker start --runtime=gpu
    ```
    Alternatively, you can run the server with CPU support by replacing `--runtime=gpu` with `--runtime=cpu`.

=== "Via Docker Compose"

    Navigate to [`examples/quickstart`](https://github.com/autonomi-ai/nos/nos/examples/quickstart) to see an example of
    {% include '../examples/quickstart/docker-compose.quickstart.yml' %}

::: nos.init
::: nos.shutdown
