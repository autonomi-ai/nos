
In this demo, we’ll use NOS to build an end-to-end semantic video search application. Let's first start the nos server.

```python
import nos

nos.init(runtime="auto")
```

#### Frame Inference

Let's embed the video frame-by-frame with NOS. We'll start by connecting a client to the NOS server:


```python
from nos.common.io.video.opencv import VideoReader
from nos.client import Client
from nos.test.utils import get_benchmark_video

# Start the client
client = Client()
client.WaitForServer()
client.IsHealthy()

# Load the video
FILENAME = get_benchmark_video()
video = VideoReader(str(FILENAME))
```

Now lets use the client to embed the video frame by frame into a stack of feature vectors. This should take a couple of minutes:


```python
import numpy as np
from PIL import Image

from nos.common import tqdm
from nos.test.utils import get_benchmark_video

# Initialize the openai/clip model as a module
clip = client.Module("openai/clip", shm=True)

# Extract features from the video on a frame-level basis
# Note: we resize the image to 224x224 before encoding
features = [
    clip.encode_image(images=Image.fromarray(img).resize((224, 224)))["embedding"] 
    for img in tqdm(video)
]

# Stack and normalize the features so that they are unit vectors
video_features = np.vstack(features)
video_features /= np.linalg.norm(video_features, axis=1, keepdims=True)
```

Let's define our search function. we'll embed the text query (using the NOS openai/clip endpoint) and dot it with the video features to generate per-frame similarity scores before returning the top result.


```python
def search_video(query: str, video_features: np.ndarray, topk: int = 3):
    """Semantic video search demo in 8 lines of code"""
    # Encode text and normalize
    text_features = clip.encode_text(texts=query)["embedding"].copy()
    text_features /= np.linalg.norm(text_features, axis=1, keepdims=True)

    # Compute the similarity between the search query and each video frame
    similarities = (video_features @ text_features.T)
    best_photo_idx = similarities.flatten().argsort()[-topk:][::-1]
    
    # Display the top k frames
    results = np.hstack([video[int(frame_id)] for frame_id in best_photo_idx])
    filepath = '_'.join(query.split(' ')) + '.png'
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
