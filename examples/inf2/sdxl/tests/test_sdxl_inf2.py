def test_sdxl_inf2():
    from models.sdxl_inf2 import StableDiffusionXLInf2
    from PIL import Image

    model = StableDiffusionXLInf2()
    prompts = "a photo of an astronaut riding a horse on mars"
    response = model(prompts=prompts, height=1024, width=1024, num_inference_steps=50)
    assert response is not None
    assert isinstance(response[0], Image.Image)
