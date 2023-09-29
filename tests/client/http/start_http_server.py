# Create a test app to run load testing against:
import uvicorn
from nos.server.http._service import app
from nos.server import InferenceServiceRuntime
from nos.logging import logger
from nos.constants import DEFAULT_GRPC_PORT

# Start the runtime from test utils
GPU_CONTAINER_NAME = "nos-inference-service-runtime-gpu-locust-test"
runtime = InferenceServiceRuntime(runtime="gpu", name=GPU_CONTAINER_NAME)

# Force stop any existing containers
try:
    runtime.stop()
except Exception:
    logger.info(f"Killing any existing container with name: {GPU_CONTAINER_NAME}")

# Start grpc server runtime (GPU)
container = runtime.start(
    ports={f"{DEFAULT_GRPC_PORT}/tcp": DEFAULT_GRPC_PORT},
    environment={
        "NOS_LOGGING_LEVEL": "DEBUG",
    },
)
assert container is not None
assert container.id is not None
status = runtime.get_container_status()
assert status is not None and status == "running"

"""
# Tear down
try:
    runtime.stop()
except Exception:
    logger.info(f"Failed to stop existing container with name: {GPU_CONTAINER_NAME}")
"""

# Start the Fast API app with uvicorn
uvicorn.run(app(), host="localhost", port=8001, workers=1, log_level="info")
