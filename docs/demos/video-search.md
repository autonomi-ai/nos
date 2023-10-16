<img src="https://drive.google.com/uc?export=view&id=1JIIlkTWa2xbft5bTpzhGK1BxYL83bJNU" width="800"/>


# 🔥 Video Search Demo
---

In this demo, we’ll use NOS to build an end-to-end semantic video search utility.


```python
from nos.test.utils import get_benchmark_video
get_benchmark_video()
FILENAME = "test_video.mp4"
```

#### Frame Inference

Let's embed the video frame by frame with NOS. We'll start by connecting a client to the NOS server:


```python
from nos.common.io.video.opencv import VideoReader
from nos.client import Client, TaskType

client = Client()
client.WaitForServer()
client.IsHealthy()
```

Now lets use the client to embed the video frame by frame into a stack of feature vectors. This should take a couple of minutes:


```python
from nos.common import tqdm
from nos.common.io.video.opencv import VideoReader
import torch
import numpy as np
from itertools import islice

images = VideoReader(FILENAME)
features = []

for img in tqdm(images):
    features.append(client.Run(TaskType.IMAGE_EMBEDDING, "openai/clip", inputs={"images" : img})['embedding'])

# normalize embeddings
video_features = torch.from_numpy(np.stack(features))
video_features /= video_features.norm(dim=-1, keepdim=True)
```

Let's define our search function. we'll embed the text query (using the NOS openai/clip endpoint) and dot it with the video features to generate per-frame similarity scores before returning the top result.


```python
from IPython.display import HTML, display
from nos.common.io import VideoReader
from PIL import Image

video = VideoReader(FILENAME)

def search_video(query: str, video_features: np.ndarray, topk: int = 3):
    """Semantic video search demo in 8 lines of code"""
    # Encode text and normalize
    with torch.inference_mode():
        text_features = client.Run(TaskType.TEXT_EMBEDDING, "openai/clip", inputs={"texts":[query]})["embedding"]
        text_features = torch.from_numpy(text_features)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    # Compute the similarity between the search query and each video frame
    similarities = (video_features @ text_features.T)
    _, best_photo_idx = similarities.topk(topk, dim=0)

    # Display the top k frames
    results = np.hstack([video[int(frame_id)] for frame_id in best_photo_idx])
    display(Image.fromarray(results).resize((600, 400)))
```




Now let's try out a few queries:


```python
search_video("bakery with bread on the shelves", video_features, topk=1)
```
![bakery](../assets/bakery_with_bread_on_the_shelves.png)

```python
search_video("red car on a street", video_features, topk=1)
```

![red car](../assets/red_car_on_a_street.png)


```python
search_video("bridge over river", video_features, topk=1)
```

![bridge](../assets/bridge_over_river.png)
