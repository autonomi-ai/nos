### 🧠 1. Fine-tuning models with NOS

In this tutorial, we're going to show how we can fine-tune models through the NOS fine-tuning API.

```python
import shutil
from pathlib import Path

from nos.client import Client, TaskType
from nos.common import TaskType
from nos.types import TrainingJobResponse
from nos.logging import logger

# Test waiting for server to start
# This call should be instantaneous as the server is already ready for the test
client = Client()
logger.debug("Waiting for server to start...")
client.WaitForServer()

logger.debug("Confirming server is healthy...")
if not client.IsHealthy():
    raise RuntimeError("NOS server is not healthy")
```


**1. Get the training schema for the task type**
```python
# Get the training schema for the task type
schema: TrainingSchema = client.GetTrainingInputSchema(task=TaskType.OBJECT_DETECTION_2D)

# Input dataset schema (TrainingSchema)
schema: TrainingSchema = dict(
    inputs=[
        dict(name="image_id", type=int,
            description="Unique image identifier for the image"),
        dict(name="image_path", type=str,
            description="Image path (relative to the dataset root)"),
        dict(name="gt_bboxes", type=np.ndarray,
            description="Ground-truth bounding boxes [(x1, y1, x2, y2), ...]"),
        dict(name="gt_labels", type=np.ndarray,
            description="Ground-truth labels [label_id, ...]"),
        dict(name="dataset_split", type=str,
            description="Split (e.g. train, val, test, etc.)"),
    ],
    overrides=...
    metadata=...,
)

# Supported dataset schemas
 - dataframe: pandas.DataFrame
    - input_uri: Union[Path, RemotePath, pd.DataFrame]
 - coco: COCO dataset schema (input_uri should be a cross-mounted directory/volume)
    - input_uri: Union[Path, RemotePath]
 - ...

```

**2. Export training dataset view**
```python
# Create a volume for training data
# Note: Volumes need to be cross-mounted on the server
volume_dir = client.Volume("datasets/coco128/<snapshot_id>")

# Export the pixeltable training data to the volume
# Note: Additional dataset schemas can be supported (e.g. COCO, VOC, etc.)
dataset_uri = pt_table.export(volume_dir, schema="dataframe")
```

**3. Fine-tune YOLO-X on exported dataset view**

```python
# Train a new YOLOX-s model on the dataset view
response: TrainingJobResponse = client.Train(
    task=TaskType.OBJECT_DETECTION_2D,
    model_name="open-mmlab/yolox-small",
    # Standard training inputs exposed by the NOS API
    inputs={
        "dataset_uri": dataset_uri,
        "dataset_schema": "dataframe",
    },
    # Additional training inputs (overridden) specific to the model
    # Note: Every config parameter can be overridden if needed.
    overrides={
        "load_from": "https://.../yolox_s-42aa3d00.pth",
        "optim_wrapper": {"optimizer": dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)},
        "max_epochs": 100,
    },
    # Additional metadata to attach to the training job
    metadata={
        "model_name": "yolox-s",
        "model_type": "torch",
        "model_version": "1.0",
    }
)
assert response is not None
job_id = response.job_id
logger.debug(f"Training job dispatched [job_id={job_id}].")

# Check status for the training job to complete
status = client.GetTrainingJobStatus(job_id)
logger.debug(f"Training job [status={status}]")

# TrainingJobResponse
# model_id       - unique model ID assigned to the trained model
# job_id         - unique job ID assigned to the training job
# experiment_uri - remote URI to the experiment statistics
# model_uri      - remote URI to the trained model
# status         - status of the training job


```
**4. Running inference with the fine-tuned model**

Once trained, models are automatically registered under the `custom/` namespace. Each fine-tuned model is assigned a unique `model_id` that can be used to retrieve the model handle.

```python
from nos.client import Client, TaskType
from nos.test.utils import NOS_TEST_IMAGE

# Test inference with the fine-tuned model
model_id = response.model_id

logger.debug(f"Testing inference using the fine-tuned model [model_id={model_id}]...")
predictions = client.Run(
    task=TaskType.OBJECT_DETECTION_2D,
    model_name=model_id,
    images=[NOS_TEST_IMAGE],
)
logger.debug(f"Detections: {predictions}")
```
