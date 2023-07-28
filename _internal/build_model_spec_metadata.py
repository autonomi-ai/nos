from nos import hub
from nos.common import TaskType
from nos.common.spec import ModelResources, ModelSpec, ModelSpecMetadata, _metadata_path
from nos.logging import logger


spec: ModelSpec = hub.load_spec(task=TaskType.IMAGE_EMBEDDING, model_name="openai/clip")
spec._metadata = ModelSpecMetadata(
    name=spec.name,
    task=spec.task,
    resources={
        "cpu": ModelResources(runtime="cpu", device="cpu", device_memory=2 * 1024**3, cpus=2),
        "gpu": ModelResources(runtime="gpu", device="cuda", device_memory=2 * 1024**3, cpus=1),
    },
)

path = _metadata_path(spec)
if not path.exists():
    path.parent.mkdir(parents=True, exist_ok=True)
    spec._metadata.to_json(path)
    logger.info(f"Saved metadata to {path}")
