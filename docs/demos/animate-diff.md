# Running animate-diff as a custom model.

[Animatediff](https://animatediff.github.io/) generates short (~1s) gifs from
text prompts. We don't support this out of the box in NOS, but in this tutorial
we'll cover how to define a configuration for a custom model with dependencies
different from those in the base NOS server image.

1. **Define a serve.yaml with required dependencies**

In `examples/animatediff` you will find two files: `serve.yaml` defines the custom image and 
model information, which in turn references `models/model.py` containing the model implementation
we will make available through NOS. The `serve.yaml` is set up as follows:
```yaml
images:
    animate-diff-gpu:
        base: autonomi/nos:latest-gpu
        pip:
        - diffusers==0.24.0
        - transformers==4.35.2
        - accelerate==0.23.0

models:
    animate-diff:
    model_cls: AnimateDiff
    model_path: models/model.py
    default_method: __call__
    runtime_env: animate-diff-gpu
```

Our custom model will run inside of `animate-diff-gpu`, which is derived from 
`latest-gpu` and adds a few huggingface packages version locked to avoid any dependency
issues for this specific model. If your model runs out of the box with the base nos dependencies then this shouldn't be necessary.

Next we define the model itself by specifying a few fields. The `model_cls` maps to the 
AnimateDiff class defined in `models/model.py`. The `model_path` should link to the model
implementation file. `default_method` will be the entrypoint that gets called when we
register a client module against our custom model id (`animate-diff`). Finally, we 
set the `runtime_env` to the image we defined above (`animate-diff-gpu`).

2. **Serving our custom model**

We can now serve the model inside of `examples/aniamtediff` with 
```bash
nos serve up -c serve.yaml
```

We're now ready to run animatediff.

3. **Generate a gif**

We create a client module as we would for any default NOS model and pass in a prompt:

```python
from nos.client import Client, TaskType

client = Client()
client.WaitForServer()
client.IsHealthy()

model = client.Module("animate-diff")
response = model(prompts=["astronaut on the moon, hdr, 4k"], _stream=True)

from PIL import Image, ImageSequence

frames = [frame for frame in response]
frames[0].save('output.gif', save_all=True, append_images=frames[1:], loop=0)
```