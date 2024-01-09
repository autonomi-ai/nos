# `nos serve` CLI

::: mkdocs-typer
    :module: nos.cli.serve
    :command: serve_cli


### Serve YAML Specification

The `serve` CLI uses a YAML specification file to fully configure the runtime and the models that need to be served. The full specification is available [here](./serve.spec.yaml). 

```yaml title="serve.yaml"
{% include './serve.spec.yaml' %}
```

Check out our custom model serving tutorial [here](../../examples/tutorials/01-define-custom-models/) to learn more about how to use the `serve.yaml` file to serve custom models with NOS.

### Debugging the server

When using the `serve` CLI, you can enable verbose logging by setting the `--logging-level` argument to `DEBUG`:

```bash
$ nos serve -c <serve.yaml> --logging-level DEBUG
```

This can be especially useful when debugging issues with your server, especially around model registry, resource allocation and model logic.