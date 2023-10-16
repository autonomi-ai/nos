NOS supports OpenAI CLIP (Contrastive Language-Image Pre-Training) out of the box for both image and text embedding. Note that the first inference call will block on model download.

```python
import nos
from nos.client import Client, TaskType
from PIL import Image
import requests

nos.init(runtime="gpu")
client = Client()
client.WaitForServer()
client.IsHealthy()

url = "https://raw.githubusercontent.com/open-mmlab/mmdetection/main/demo/demo.jpg"
img = Image.open(requests.get(url, stream=True).raw).resize((640, 480))

# single image
predictions = client.Run("openai/clip", inputs={"images": [img]}, method="image_embedding")
print(predictions["embedding"].shape) # 1X512

# batched N=3
predictions = client.Run("openai/clip", inputs={"images": [img, img, img]}, method="image_embedding")
print(predictions["embedding"].shape) # 3X512
```

Text embeddings follow a similar pattern:

```python
text_string = "the quick brown fox jumped over the lazy dog"
predictions = client.Run("openai/clip", inputs={"texts": [text_string]}, method="text_embedding")
predictions["embedding"].shape

```
