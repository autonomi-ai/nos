NOS Supports a variety of YOLOX variants for object detection:

```python
import nos
from nos.client import Client, TaskType
from PIL import Image
import requests
import cv2
import numpy as np

nos.init(runtime="gpu")
client = Client()
client.WaitForServer()
client.IsHealthy()

url = "https://raw.githubusercontent.com/open-mmlab/mmdetection/main/demo/demo.jpg"
img = Image.open(requests.get(url, stream=True).raw).resize((640, 480))

def visualize_det2d(img: np.ndarray, bboxes: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Visualize 2D detection results on an image."""
    vis = np.asarray(img).copy()
    for bbox, label in zip(bboxes.astype(np.int32), labels):
        cv2.rectangle(vis, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
    return vis

# Run YOLOX prediction on the image and get the prediction results as a dictionary.
# predictions = {"bboxes", "scores", "labels"}.
predictions = client.Run(TaskType.OBJECT_DETECTION_2D, "yolox/nano",
                         inputs={"images": [img]})
for idx, (img, bboxes, scores, labels) in enumerate(zip([img], predictions["bboxes"], predictions["scores"], predictions["scores"])):
    display(Image.fromarray(visualize_det2d(img, bboxes, labels)))
```

![Detections](../assets/bench_park_detections.png)
