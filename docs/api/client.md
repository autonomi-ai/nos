## NOS Inference Service
::: nos.init
::: nos.shutdown

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
