import pytest


@pytest.mark.parametrize("model_id", ["BAAI/bge-small-en-v1.5"])
def test_embeddings_inf2_client(model_id):
    import numpy as np

    from nos.client import Client

    # Create a client
    client = Client("[::]:50051")
    assert client.WaitForServer()

    # Load the embeddings model
    model = client.Module(model_id)

    # Embed text with the model
    texts = "What is the meaning of life?"
    response = model(texts=texts)
    assert response is not None
    assert isinstance(response, np.ndarray)
