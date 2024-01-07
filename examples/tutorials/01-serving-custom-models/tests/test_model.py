from nos.client import Client


def test_lcm():
    from typing import List

    client = Client("[::]:50051")
    assert client is not None
    assert client.WaitForServer()
    assert client.IsHealthy()

    model_id = "custom/custom-model"
    models: List[str] = client.ListModels()
    assert model_id in models

    model = client.Module(model_id)
    assert model is not None
    assert model.GetModelInfo() is not None
    print(f"Test [model={model_id}]")

    response: List[str] = model(prompts=["astronaut on the moon, hdr, 4k"])
    assert isinstance(response, list)
    print(response)
