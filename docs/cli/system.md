# `nos system` CLI

NOS provides some basic utilities for listing and inspecting your system information.

::: mkdocs-typer
    :module: nos.cli.system
    :command: system_cli

### Get system information

```
nos system info
╭─────────────────────────────────── System ───────────────────────────────────╮
│ {                                                                            │
│   "system": {                                                                │
│     "system": "Linux",                                                       │
│     "release": "5.19.0-41-generic",                                          │
│     "version": "#42~22.04.1-Ubuntu SMP PREEMPT_DYNAMIC Tue Apr 18 17:40:00 U │
│     "machine": "x86_64",                                                     │
│     "architecture": [                                                        │
│       "64bit",                                                               │
│       "ELF"                                                                  │
│     ],                                                                       │
│     "processor": "x86_64",                                                   │
│     "python_implementation": "CPython"                                       │
│   },                                                                         │
│   "cpu": {                                                                   │
│     "model": "AMD Ryzen Threadripper 3970X 32-Core Processor",               │
│     "architecture": "x86_64",                                                │
│     "cores": {                                                               │
│       "physical": 32,                                                        │
│       "total": 64                                                            │
│     },                                                                       │
│     "frequency": 3300.0,                                                     │
│     "frequency_str": "3.30 GHz"                                              │
│   },                                                                         │
│   "memory": {                                                                │
│     "total": 134905909248,                                                   │
│     "used": 9529114624,                                                      │
│     "available": 119143944192                                                │
│   },                                                                         │
│   "torch": {                                                                 │
│     "version": "2.0.1"                                                       │
│   },                                                                         │
│   "docker": {                                                                │
│     "version": "Docker version 24.0.0, build 98fdcd7",                       │
│     "sdk_version": "6.1.0",                                                  │
│     "compose_version": "Docker Compose version v2.17.3"                      │
│   },                                                                         │
│   "gpu": {                                                                   │
│     "cuda_version": "11.7",                                                  │
│     "cudnn_version": 8500,                                                   │
│     "device_count": 3,                                                       │
│     "devices": [                                                             │
│       {                                                                      │
│         "device_id": 0,                                                      │
│         "device_name": "NVIDIA GeForce RTX 4090",                            │
│         "device_capability": "8.9",                                          │
│         "total_memory": 25393692672,                                         │
│         "total_memory_str": "23.65 GB",                                      │
│         "multi_processor_count": 128                                         │
│       },                                                                     │
│       {                                                                      │
│         "device_id": 1,                                                      │
│         "device_name": "NVIDIA GeForce RTX 2080 Ti",                         │
│         "device_capability": "7.5",                                          │
│         "total_memory": 11543379968,                                         │
│         "total_memory_str": "10.75 GB",                                      │
│         "multi_processor_count": 68                                          │
│       },                                                                     │
│       {                                                                      │
│         "device_id": 2,                                                      │
│         "device_name": "NVIDIA GeForce RTX 2080 Ti",                         │
│         "device_capability": "7.5",                                          │
│         "total_memory": 11546394624,                                         │
│         "total_memory_str": "10.75 GB",                                      │
│         "multi_processor_count": 68                                          │
│       }                                                                      │
│     ],                                                                       │
│     "driver_version": "530.41.03"                                            │
│   }                                                                          │
│ }                                                                            │
╰──────────────────────────────────────────────────────────────────────────────╯
GPU detected, fetching nvidia-smi information within docker.
...
```
