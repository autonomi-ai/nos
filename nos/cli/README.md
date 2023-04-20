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
