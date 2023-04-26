# NOS CLI

The `nos` CLI is the main entrypoint for running NOS from the command-line. It is designed to be easy to use and extend.

Commands are organized into subcommands, which are organized into groups. For example, the `nos hub info` command is part of the `nos hub` group.

## Subcommands

| Section                     | Subcommand      | Description                                                                           |
|:----------------------------|:----------------|:--------------------------------------------------------------------------------------|
| [system](./system.py)       | `nos system`    | System-related commands to get information about your system.                         |
| [docker](./docker.py)       | `nos docker`    | Docker-related commands to pull, initialize and run the nos docker runtime container. |
| [hub](./hub.py)             | `nos hub`       | Hub-related commands to pull, push, and manage ML models.                             |
| [benchmark](./benchmark.py) | `nos benchmark` | Benchmark-related commands to run benchmarks on your models.                          |
| [serve](./serve.py)         | `nos serve`     | ML model serving related comamnds serve your ML models via a REST/gRPC API.           |

## Examples

```bash
# List all available models in the hub
nos hub list

# Download model and compile/archive it for inference
nos hub download -m stabilityai/stable-diffusion-2

# Optimize model for inference (WIP)
nos opt optimize -m stabilityai/stable-diffusion-2

# Serve the model for inference (WIP)
nos serve -m stabilityai/stable-diffusion-2 --device auto
```


## Getting Started

To get started, install the `nos` CLI using `pip`:

```bash
pip install nos
```

## Demo

Let's walk through a demo of how to use the `nos` CLI to download, optimize, and serve a model.

Currently supported models include:
 - `openai/clip-*`
 - `stabilityai/stable-diffusion-*`

Deploy the model using the following command:
```bash
nos serve deploy -m stabilityai/stable-diffusion-2 --min-replicas 1
```

On the client side, you can use the following command to query the model:
```bash
nos serve predict -m txt2img -i 'overhead imagery of trees'
```
