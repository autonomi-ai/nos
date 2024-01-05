from typing import List

import pytest

from nos.client import Client


@pytest.fixture(scope="session", autouse=True)
def client():
    # Create a client
    client = Client("[::]:50051")
    assert client is not None
    assert client.WaitForServer()
    yield client


def test_custom_methods(client):
    model_id = "custom/custom-model"
    models: List[str] = client.ListModels()
    assert model_id in models

    model = client.Module(model_id)
    assert model is not None
    assert model.GetModelInfo() is not None
    print(f"Test [model={model_id}]")

    # Test the default __call__ method
    response = model(arg1="Hello World!")
    assert isinstance(response, str)

    # Test method1
    response = model.method1(arg1="Hello World!")
    assert isinstance(response, str)

    # Test method2
    response = model.method2(arg1="Hello World!")
    assert isinstance(response, str)

    # Test methods (with _method kwarg)
    response = model(arg1="Hello World!", _method="method1")
    assert isinstance(response, str)
    response = model(arg1="Hello World!", _method="method2")
    assert isinstance(response, str)
