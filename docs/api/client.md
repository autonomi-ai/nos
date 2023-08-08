## NOS Inference Service

In this section, we expect that you have already installed NOS and have already [started the server](/docs/starting-the-server).

## ::: nos.client.grpc.InferenceClient
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

## ::: nos.client.grpc.InferenceModule
    handler: python
    options:
      members:
        - __init__
        - GetModelInfo
        - __call__
