import pytest

from nos.test.conftest import grpc_client  # noqa: F401


@pytest.mark.client
def test_client_cloudpickle_serialization(grpc_client):  # noqa: F811
    """Test cloudpickle serialization."""
    import cloudpickle

    stub = grpc_client.stub  # noqa: F841

    def predict_wrap():
        return grpc_client.Predict(
            method="txt2vec",
            model_name="openai/clip-vit-base-patch32",
            text="This is a test",
        )

    predict_fn = cloudpickle.dumps(predict_wrap)
    assert isinstance(predict_fn, bytes)
