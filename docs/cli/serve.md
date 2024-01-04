# `nos serve` CLI

::: mkdocs-typer
    :module: nos.cli.serve
    :command: serve_cli


### Serve YAML Specification

The `serve` CLI takes a YAML specification file that defines the server configuration. 

```yaml title="serve.yaml"
{% include './serve.spec.yaml' %}
```

### Debugging the server

When using the `serve` CLI, you can enable verbose logging by setting the `--logging-level` argument to `DEBUG`:

```bash
$ nos serve -c <serve.yaml> --logging-level DEBUG
```

This can be especially useful when debugging issues with your server, especially around model registry, resource allocation and model logic.