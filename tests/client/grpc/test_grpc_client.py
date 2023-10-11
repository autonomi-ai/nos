import pytest

from nos.common import TaskType
from nos.test.conftest import grpc_client  # noqa: F401


@pytest.mark.client
def test_client_cloudpickle_serialization(grpc_client):  # noqa: F811
    """Test cloudpickle serialization."""
    from nos.common.cloudpickle import dumps

    stub = grpc_client.stub  # noqa: F841

    def predict_wrap():
        return grpc_client.Run(
            task=TaskType.IMAGE_EMBEDDING,
            model_name="openai/clip-vit-base-patch32",
            inputs={"texts": "This is a test"},
        )

    predict_fn = dumps(predict_wrap)
    assert isinstance(predict_fn, bytes)

    def predict_module_wrap():
        module = grpc_client.Module(task=TaskType.IMAGE_EMBEDDING, model_name="openai/clip-vit-base-patch32")
        return module(inputs={"prompts": ["This is a test"]})

    predict_fn = dumps(predict_module_wrap)
    assert isinstance(predict_fn, bytes)

    def train_wrap():
        return grpc_client.Train(
            method="stable-diffusion-dreambooth-lora",
            inputs={
                "model_name": "stabilityai/stable-diffusion-2-1",
                "instance_directory": "/tmp",
                "instance_prompt": "A photo of a bench on the moon",
            },
            metadata={
                "name": "sdv21-dreambooth-lora-test-bench",
            },
        )

    train_fn = dumps(train_wrap)
    assert isinstance(train_fn, bytes)
