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
            texts="This is a test",
        )

    predict_fn = dumps(predict_wrap)
    assert isinstance(predict_fn, bytes)

    def predict_module_wrap():
        module = grpc_client.Module(task=TaskType.IMAGE_EMBEDDING, model_name="openai/clip-vit-base-patch32")
        return module(prompts=["This is a test"])

    predict_fn = dumps(predict_module_wrap)
    assert isinstance(predict_fn, bytes)
