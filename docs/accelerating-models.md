### 🔥 Inference with HuggingFace model

We're going to accelerate the popular OpenAI CLIP model for image-embeddings using 🤗 **transformers** `CLIPModel`.

Let's define a custom `CustomCLIPModel`.

```python
from typing import Union, List
from PIL import Image

import numpy as np
import torch

class CustomCLIPModel:
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        from transformers import CLIPModel, CLIPProcessor, CLIPTokenizer

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

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

---
### 🚀 2. Optimizing Inference with NOS

NOS provides a convenient way to compile, optimize and auto-scale the model for inference.

#### 🔌 2a. Connect to the NOS server

Now, let’s start the nos runtime. NOS can sit as a docker service running locally or in the cloud accessing 100s of GPUs in a cluster.

```python
import nos
from nos.managers.model import ModelManager, ModelOptimizationPolicy

nos.init(runtime="local")
```

#### ⚡️ 2b. Accelerating a Custom Pytorch Model

Let's say you want to accelerate the `CustomCLIPModel` Pytorch model we just used.

```python
# Initialize the model manager
manager = ModelManager()

# Trace the model and build the model spec
spec = nos.trace(
    CustomCLIPModel,
    init_args=(), init_kwargs={"model_name": "openai/clip-vit-base-patch32"},
    method_name="encode_image")

# Load the model from the spec
model = manager.load(spec)
```

Let's see what the model looks like:

```bash
ModelHandle(name=CustomCLIPModel, replicas=1, qsize=2, opts=(num_gpus=0.1))
```

Now, let's optimize the model!

```python
# Optimize the model for maximum throughput
model = model.optimize(policy=ModelOptimizationPolicy.MAX_THROUGHPUT)
```

NOS automatically decides the optimal number of replicas to give us the best performance for the hardware we have.

```bash
ModelHandle(name=CustomCLIPModel, replicas=1, qsize=2, opts=(num_gpus=0.1))
    device: NVIDIA GeForce RTX 4090 (0)
    optimal: mem_usage=2.0 GB, num_replicas=8, batch_size=16, throughput=1036.5 im/s, latency=15.7 ms
```

---
### 🔥 Accelerating a vanilla Pytorch Model with `nos.optimize`

We take the model we just optimized, and scale this up! NOS automatically decides the optimal number of replicas to give us the best performance for the hardware we have.

```python
 model = model.scale(replicas="auto")
```

Now, if we check the model again, we can see that NOS has scaled the model to 8 replicas.

```bash
ModelHandle(name=CustomCLIPModel, replicas=8, qsize=16, opts=(num_gpus=0.1))
    device: NVIDIA GeForce RTX 4090 (0)
    optimal: mem_usage=2.0 GB, num_replicas=8, batch_size=16, throughput=1036.5 im/s, latency=15.7 ms
```

Finally let’s take this optimized and scaled model to build our video search engine.

```python
model.imap(images)  # batched inference
```
