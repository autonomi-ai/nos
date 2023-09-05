### 🧠 1. Fine-tuning models with NOS

In this tutorial, we're going to show how we can fine-tune models through the NOS fine-tuning API.

```python
import shutil
from pathlib import Path

from nos.common import TaskType
from nos.types import TrainingJobResponse
from nos.logging import logger
from nos.test.utils import NOS_TEST_IMAGE

# Test waiting for server to start
# This call should be instantaneous as the server is already ready for the test
client = InferenceClient()
logger.debug("Waiting for server to start...")
client.WaitForServer()

logger.debug("Confirming server is healthy...")
if not client.IsHealthy():
    raise RuntimeError("NOS server is not healthy")

# Create a volume for training data
# Note: Volumes need to be cross-mounted on the server
volume_dir = client.Volume("datasets/coco128/<snapshot_id>")

# Export the pixeltable training data to the volume
# Note: Additional dataset schemas can be supported (e.g. COCO, VOC, etc.)
pt_table.save(volume_dir, schema="pixeltable")

# Train a new YOLOX-s model on the dataset view
response: TrainingJobResponse = client.Train(
    method="openmmlab/mmdetection",
    # Standard training inputs exposed by the NOS API
    inputs={
        "model_name": "open-mmlab/yolox-small",
        "input_directory": volume_dir,
        "dataset_schema": "pixeltable",
    },
    # Additional training inputs (overridden) specific to the model
    # Note: Every config parameter can be overridden if needed.
    overrides={
        "load_from": "https://.../yolox_s-42aa3d00.pth",
        "optimizer": dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
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

logger.debug(f"Training service dispatched [model_id={model_id}].")

```

---
### 🚀 2. Running inference with fine-tuned models

Once trained, models are automatically registered under the `custom/` namespace. Each fine-tuned model is assigned a unique `model_id` that can be used to retrieve the model handle.

```python
from nos.client import InferenceClient, TaskType
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
