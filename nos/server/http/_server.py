from typing import Any, Dict, Optional

from fastapi import Depends, FastAPI, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from fastapi import File, UploadFile, status

from nos.client import DEFAULT_GRPC_PORT, InferenceClient
from nos.common import TaskType
from nos.logging import logger
from nos.protoc import import_module
from nos.server.http._utils import encode_dict, decode_file_object
from nos.version import __version__


class InferenceRequest(BaseModel):
    task: str
    """Task used for inference"""
    model_name: str
    """Model identifier"""
    inputs: Dict[str, Any]
    """Input data for inference"""
    data: Optional[UploadFile] = File(None)
    """Uploaded image / video / audio file for inference"""


nos_service_pb2 = import_module("nos_service_pb2")
nos_service_pb2_grpc = import_module("nos_service_pb2_grpc")

API_VERSION = "v1"
app = FastAPI(
    title="NOS REST API",
    description=f"NOS REST API Server Backend (version={__version__}, api_version={API_VERSION})",
    version=API_VERSION,
    debug=True,
)
logger.debug(f"Starting NOS REST API Server Backend (version={__version__})")

client = InferenceClient(f"[::]:{DEFAULT_GRPC_PORT}")
logger.debug(f"Connecting to gRPC server at {client.address}")

client.WaitForServer(timeout=60)
runtime = client.GetServiceRuntime()
version = client.GetServiceVersion()
logger.debug(f"Connected to gRPC server (runtime={runtime}, version={version})")


def get_client() -> InferenceClient:
    """Get the inference client."""
    return client


@app.get(f"/{API_VERSION}/ping", status_code=status.HTTP_200_OK)
def ping(client=Depends(get_client)) -> JSONResponse:
    """Check if the server is alive."""
    return JSONResponse(content={"status": "ok" if client.IsHealthy() else "not_ok"}, status_code=status.HTTP_200_OK)


@app.post(f"/{API_VERSION}/infer", status_code=status.HTTP_201_CREATED)
def infer(request: InferenceRequest, 
          client=Depends(get_client)) -> JSONResponse:
    """Perform inference on the given input data.
    
    $ curl -X "POST" \
      "http://localhost:8000/v1/infer" \
      -H "Content-Type: multipart/form-data" \
      -F request='{"task": "image_classification", "model_name": "yolox/small"}' \
      -F data=@/path/to/image.jpg;type=image/jpeg

    Args:
        request: Inference request.
        file_object: Uploaded image / video / audio file for inference.
        client: Inference client.

    Returns:
        Inference response.
    """
    # request = InferenceRequest(**request.dict())
    try:
        task: TaskType = TaskType(request.task)
    except KeyError:
        logger.error(f"Task '{request.task}' not supported")
        return JSONResponse(
            content={"error": f"Task '{request.task}' not supported"}, status_code=status.HTTP_400_BAD_REQUEST
        )

    try:
        model = client.Module(task=task, model_name=request.model_name)
    except Exception:
        logger.error(f"Model '{request.model_name}' not supported")
        return JSONResponse(
            content={"error": f"Model '{request.model_name}' not supported"}, status_code=status.HTTP_400_BAD_REQUEST
        )

    if request.data is not None:
        logger.debug(f"Decoding file object {request.data}")
        inputs: Dict[str, Any] = decode_file_object(request.data)
        inputs = {**inputs, **request.inputs}
    else:
        inputs = request.inputs
    logger.debug(f"Inference [task={task}, model_name={request.model_name}, keys={inputs.keys()}]")
    response = model(**inputs)
    logger.debug(f"Inference [task={task}, model_name={request.model_name}, response={response}]")
    return JSONResponse(content=encode_dict(response), status_code=status.HTTP_201_CREATED)