## NOS Inference Service

In this section, we expect that you have already installed NOS and have already [started the server](../starting-the-server.md).

## ::: nos.client.grpc.Client
    handler: python
    options:
      members:
        - __init__
        - IsHealthy
        - WaitForServer
        - GetServiceVersion
        - CheckCompatibility
        - ListModels
        - GetModelInfo
        - Module
        - ModuleFromSpec
        - Run

## ::: nos.client.grpc.Module
    handler: python
    options:
      members:
        - __init__
        - GetModelInfo
        - __call__
