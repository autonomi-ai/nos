from nos import hub
from nos.common import TaskType
from nos.common.spec import ModelResources, ModelSpec, ModelSpecMetadata


# path = NOS_PATH / f"data/models/{id}/metadata.json"
spec: ModelSpec = hub.load_spec(task=TaskType.IMAGE_EMBEDDING, model_name="openai/clip")
spec._metadata = ModelSpecMetadata(
    name=spec.name,
    task=spec.task,
    runtime={"cpu", "gpu", "trt"},
    resources=ModelResources(device="cuda", device_memory=2 * 1024**3, cpus=1),
)
import pdb; pdb.set_trace()
