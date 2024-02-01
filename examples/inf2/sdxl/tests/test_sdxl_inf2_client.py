import pytest


@pytest.mark.parametrize("model_id", ["stabilityai/stable-diffusion-xl-base-1.0-inf2"])
def test_sdxl_inf2_client(model_id):
    from PIL import Image

    from nos.client import Client

    # Create a client
    client = Client("[::]:50051")
    assert client.WaitForServer()

    # Load the embeddings model
    model = client.Module(model_id)

    # Run inference
    prompts = "a photo of an astronaut riding a horse on mars"
    response = model(prompts=prompts, height=1024, width=1024, num_inference_steps=50)
    assert response is not None
    assert isinstance(response[0], Image.Image)
