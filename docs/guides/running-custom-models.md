!!!info "Advanced topic"
    This guide is for advanced users of the NOS server-side custom model registry. If you're looking for a way to quickly define your custom model and runtime for serving purposes, we recommend you go through the [serving custom models](./serving-custom-models.md) guide first. 

In this guide, we will walk through how to run custom models with NOS. We will use the [OpenAI CLIP](https://huggingface.co/openai/clip-vit-base-patch32) model from the popular [HuggingFace](https://huggingface.co/) library to load the model, and then use `nos` to wrap and execute the model at scale.

## 👩‍💻 Defining the custom model

Here we're using the popular OpenAI CLIP for extracting embeddings using the Huggingface `transformers` `CLIPModel`.

```python linenums="1"
from typing import Union, List
from PIL import Image

import numpy as np
import torch

class CLIP:
    """Text and image encoding using OpenAI CLIP"""
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        from transformers import CLIPModel
        from transformers import CLIPProcessor, CLIPTokenizer

        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained(model_name).to(device)
        self.model.eval()
        self.device = self.model.device

    def encode_image(self, images: Union[Image.Image, np.ndarray, List[Image.Image], List[np.ndarray]]):
        """Encode image into an embedding."""
        with torch.inference_mode():
            inputs = self.processor(images=images, return_tensors="pt").to(self.device)
            return self.model.get_image_features(**inputs).cpu().numpy()

    def encode_text(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Encode text into an embedding."""
        with torch.inference_mode():
            if isinstance(texts, str):
                texts = [texts]
            inputs = self.tokenizer(
                texts,
                padding=True,
                return_tensors="pt",
            ).to(self.device)
            text_features = self.model.get_text_features(**inputs)
            return text_features.cpu().numpy()
```

## 📦 Wrapping the custom model

In the section below, we'll show you a straightforward way to wrap the CLIP model with `nos` and run it at scale. In theory, you can wrap any custom Python class that is serializable with `cloudpickle`. Models are wrapped with the [`ModelSpec`](../api/common/spec.md#nos.common.spec.ModelSpec) class, which is a serializable specification of a model. In this example, we'll use the [`ModelSpec.from_cls`](../api/common/spec.md#nos.common.spec.ModelSpec.from_cls) method to wrap the CLIP model.

```python linenums="1"

from nos.common import ModelSpec
from nos.manager import ModelManager, ModelHandle

# Create a model spec from the CLIP class
spec = ModelSpec.from_cls(
    CLIP,
    init_args=(),
    init_kwargs={"model_name": "openai/clip-vit-base-patch32"},
)

# Create a model manager to load the CLIP model
handle: ModelHandle = manager.load(spec)

# Encode images just like using custom methods `encode_image`
img_embedding = handle.encode_image(images=[img])

# Encode text just like using custom methods `encode_text`
txt_embedding = handle.encode_text(texts=["fox jumped over the moon"])
```

As you can see, we can use the `ModelHandle` to call the underlying methods `encode_image` and `encode_text` just like we would with the original `CLIP` class. The `ModelHandle` is a **logical handle** for the model that allows us to run the model at scale without having to worry about the underlying details of the model.

## 🚀 Scaling the model

Once the model handle has been created, we can also use it to scale the model across multiple GPUs, or even multiple nodes. `ModelHandle` exposes a [`scale()`](../api/managers.md#nos.managers.model.ModelHandle.scale) method that allows you to manually specify the number of replicas to scale the model. Optionally, you can also specify a more advanced NOS feature where the number of replicas is automatically inferred based on the memory overhead of the model via `scale(replicas="auto")`.

We continue considering the example above and scale the model to 4 replicas. In order to use all the underlying replicas effectively, we need to ensure that the calls to the underlying methods `encode_image` and `encode_text` are no longer blocking. In other words, we need to ensure that the calls to the underlying methods are asynchronous and can fully utilize the model replicas without blocking on each other. NOS provides a few convenience methods to `submit` tasks and retrieve results asynchronously using it's `handle.results` API.

```python linenums="1"
# Scale the above model handle to 4 replicas
handle.scale(replicas=4)
print(handle)

# Asynchronously encode images using the `encode_image.submit()`.
# Every custom method is patched with a `submit()` method that allows you to asynchronously
# submit tasks to the underlying model replicas.
def encode_images_imap(images_generator):
    # Iterate over the images generator
    for images in images_generator:
        # Submit the task to the underlying model replicas
        handle.encode_image.submit(images=images)
        # Wait for the results to be ready before yielding the next batch
        if handle.results.ready():
            yield handle.results.get_next()
    # Yeild all the remaining results
    while not handle.results.ready():
        yield handle.results.get_next()

images_generator = VideoReader(FILENAME)
# Encode images asynchronously
for embedding in encode_images_imap(images_generator=images):
    # Do something with the image embeddings
```

In the example above, we load images from a video file and asynchronously submit `encode_image` tasks to the 4 replicas we trivially created using the `handle.scale(replicas=4)` call. We showed how you could implement a strawman, yet performant `imap` implementation that asynchronously submits tasks to the underlying replicas and yields the results as they become available. This allows us to fully utilize the underlying replicas without blocking on each other, and thereby fully utilizing the underlying hardware.

## 🛠️ Running models in a custom runtime environment

For custom models that require execution in a custom runtime environment (e.g. with `TensorRT` or other library dependencies), we can specify the runtime environment via the `runtime_env` argument in the `ModelSpec`.

```python linenums="1"
class CustomModel:
    """Custom inference model with scikit-learn."""

    def __init__(self, model_name: str = "fake_model"):
        """Initialize the model."""
        import sklearn  # noqa: F401

    def __call__(self, images: Union[np.ndarray, List[np.ndarray]], n: int = 1) -> np.ndarray:
        ...


# Create a model spec with a custom runtime environment (i.e. with scikit-learn installed)
spec = ModelSpec.from_cls(
    CustomModel,
    init_args=(),
    init_kwargs={"model_name": "fake_model"},
    runtime_env=RuntimeEnv.from_packages(["scikit-learn"]),
)
```

For more details about custom runtime environments, please see the [runtime environments](../concepts/runtime-environments.md) section.
