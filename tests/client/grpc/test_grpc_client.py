def test_client_cloudpickle_serialization(test_grpc_client):
    """Test cloudpickle serialization."""
    import cloudpickle

    stub = test_grpc_client.stub  # noqa: F841

    def predict_wrap():
        return test_grpc_client.Predict(
            method="txt2vec",
            model_name="openai/clip-vit-base-patch32",
            text="This is a test",
        )

    predict_fn = cloudpickle.dumps(predict_wrap)
    assert isinstance(predict_fn, bytes)
