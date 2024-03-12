import numpy as np


def test_embeddings_inf2():
    from models.embeddings_inf2 import EmbeddingServiceInf2

    model = EmbeddingServiceInf2()
    texts = "What is the meaning of life?"
    response = model(texts=texts)
    assert response is not None
    assert isinstance(response, np.ndarray)
    print(response.shape)
