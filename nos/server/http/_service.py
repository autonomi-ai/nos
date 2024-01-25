import contextlib
import dataclasses
import os
import time
from dataclasses import field
from functools import lru_cache
from pathlib import Path
from tempfile import NamedTemporaryFile, SpooledTemporaryFile
from typing import Any, Dict, List, Optional

import requests
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse, StreamingResponse
from PIL import Image
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

from nos.client import Client
from nos.common.tasks import TaskType
from nos.constants import DEFAULT_GRPC_ADDRESS
from nos.logging import logger
from nos.protoc import import_module
from nos.version import __version__

from ._utils import decode_item, encode_item
from .integrations.openai.models import (
    ChatCompletionsRequest,
    ChatModel,
    Choice,
    Completion,
    DeltaChoice,
    DeltaContent,
    DeltaEOS,
    DeltaRole,
    Message,
    Model,
)


HTTP_API_VERSION = "v1"
HTTP_ENV = os.getenv("NOS_HTTP_ENV", "prod")

nos_service_pb2 = import_module("nos_service_pb2")
nos_service_pb2_grpc = import_module("nos_service_pb2_grpc")


@dataclass
class InferenceRequest:
    model_id: str
    """Model identifier"""
    inputs: Dict[str, Any]
    """Input data for inference"""
    method: str = field(default=None)
    """Inference method"""
    stream: bool = field(default=False)
    """Whether to stream the response"""


@dataclasses.dataclass
class InferenceService:
    """HTTP server application for NOS API."""

    version: str = field(default="v1")
    """NOS version."""

    address: str = field(default=DEFAULT_GRPC_ADDRESS)
    """gRPC address."""

    env: str = field(default=HTTP_ENV)
    """Environment (dev/prod/test)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    """Model configuration."""

    def __post_init__(self):
        """Post initialization."""
        self.app = FastAPI(
            title="NOS REST API",
            description=f"NOS REST API (version={__version__}, api_version={self.version})",
            version=self.version,
            debug=self.env != "prod",
        )
        logger.debug(f"Starting NOS REST API (version={__version__}, env={self.env})")

        self.client = Client(self.address)
        logger.debug(f"Connecting to gRPC server (address={self.client.address})")

        if not self.client.WaitForServer(timeout=60, retry_interval=5):
            raise RuntimeError("Failed to connect to gRPC server")
        if not self.client.IsHealthy():
            raise RuntimeError("gRPC server is not healthy")
        runtime = self.client.GetServiceRuntime()
        version = self.client.GetServiceVersion()
        logger.debug(f"Connected to gRPC server (address={self.client.address}, runtime={runtime}, version={version})")


def is_image(filename: str) -> bool:
    """Check if the given filename is an image."""
    return filename.lower().endswith((".jpg", ".jpeg", ".png"))


def is_video(filename: str) -> bool:
    """Check if the given filename is a video."""
    return filename.lower().endswith((".mp4", ".avi", ".mov"))


def is_audio(filename: str) -> bool:
    """Check if the given filename is an audio."""
    return filename.lower().endswith((".mp3", ".mp4", ".wav", ".flac", ".m4a", ".webm"))


@contextlib.contextmanager
def as_path(file: SpooledTemporaryFile, suffix: str, chunk_size_mb: int = 4 * 1024 * 1024) -> Path:
    """Save an in-memory SpooledTemporaryFile to a temporary file on the
    file-system and yield its path.

    Args:
        file (SpooledTemporaryFile): In-memory SpooledTemporaryFile to save (temporarily).
        suffix (str): File suffix (mp3, avi etc).
        chunk_size_mb (int): Chunk size for reading the file. Defaults to 4MB.
    Yield:
        (Path) Path to the temporary file.
    """
    with NamedTemporaryFile(suffix=suffix) as tmp:
        path = Path(tmp.name)
        file.seek(0)
        for chunk in iter(lambda: file.read(chunk_size_mb), b""):
            tmp.write(chunk)
        tmp.flush()
        yield path


_model_table: Dict[str, ChatModel] = {}


def app_factory(version: str = HTTP_API_VERSION, address: str = DEFAULT_GRPC_ADDRESS, env: str = HTTP_ENV) -> FastAPI:
    """Create a FastAPI factory application for the NOS REST API gateway.

    Args:
        version (str): NOS version.
        address (str): gRPC address.
        env (str): Environment (prod, dev, test).

    Returns:
        (FastAPI) FastAPI application.
    """
    nos_app = InferenceService(version=version, address=address, env=env)
    app = nos_app.app

    def get_client() -> Client:
        """Get the inference client."""
        return nos_app.client

    def unnormalize_id(model_id: str) -> str:
        """Un-normalize the model identifier."""
        return model_id.replace("--", "/")

    def normalize_id(model_id: str) -> str:
        """Normalize the model identifier."""
        return model_id.replace("/", "--")

    @lru_cache(maxsize=1)
    def build_model_table(client: Client) -> Dict[str, ChatModel]:
        """Build the model table."""
        if len(_model_table) > 0:
            return _model_table

        models: List[str] = client.ListModels()
        for model_id in models:
            spec = client.GetModelInfo(model_id)
            if not (spec.task() == TaskType.TEXT_GENERATION or spec.task() == TaskType.CUSTOM):
                continue
            try:
                owned_by, _ = model_id.split("/")
            except ValueError:
                owned_by = "unknown-org"
            _model_table[model_id] = ChatModel(id=normalize_id(model_id), created=0, owned_by=owned_by)
            logger.debug(f"Registered model [model={model_id}, m={_model_table[model_id]}, spec={spec}]")
        return _model_table

    @app.get("/")
    def root():
        return {"Hello": "World"}

    @app.get(f"/{version}/health", status_code=status.HTTP_200_OK)
    def health(client: Client = Depends(get_client)) -> JSONResponse:
        """Check if the server is alive."""
        return JSONResponse(
            content={"status": "ok" if client.IsHealthy() else "not_ok"}, status_code=status.HTTP_200_OK
        )

    @app.get(f"/{version}/models", status_code=status.HTTP_200_OK, response_model=Model)
    def models(
        client: Client = Depends(get_client),
    ) -> Model:
        """List all available models."""
        _model_table = build_model_table(client)
        logger.debug(f"Listing models [models={_model_table.values()}]")
        return Model(data=list(_model_table.values()))

    @app.get(f"/{version}/models/" + "{model:path}", response_model=ChatModel)
    def model_info(model: str, client: Client = Depends(get_client)) -> ChatModel:
        """Get model information."""
        _model_table = build_model_table(client)
        try:
            return _model_table[unnormalize_id(model)]
        except KeyError:
            raise HTTPException(status_code=400, detail=f"Invalid model {model}")

    @app.post(f"/{version}/chat/completions", status_code=status.HTTP_201_CREATED)
    def chat(
        request: ChatCompletionsRequest,
        client: Client = Depends(get_client),
    ) -> StreamingResponse:
        """Perform chat completion on the given input data."""
        logger.debug(f"Received chat request [model={request.model}, messages={request.messages}]")
        _model_table = build_model_table(client)
        model_id: str = unnormalize_id(request.model)
        try:
            _ = _model_table[model_id]
        except KeyError:
            raise HTTPException(status_code=400, detail=f"Invalid model {request.model}")
        model = client.Module(model_id)

        if not len(request.messages):
            raise HTTPException(status_code=400, detail="Invalid chat request, no messages provided")

        if len(request.messages) > 0 and request.messages[-1].role != "user":
            raise HTTPException(status_code=400, detail="Invalid chat request, last message must be from the user")

        # Perform chat completion (streaming)
        messages = [message.dict() for message in request.messages]
        if request.stream:

            def openai_streaming_generator():
                """Streaming generator for OpenAI chat completion."""
                # Add responses incrementally to the chat
                choices = [DeltaChoice(delta=DeltaRole(content="", role="assistant"), index=0, finish_reason=None)]
                yield f"data: {Completion(id=request.id, object='chat.completion.chunk', model=request.model, choices=choices).json()}\n\n"
                for response in model.chat(
                    messages=messages,
                    max_new_tokens=request.max_tokens,
                    _stream=True,
                ):
                    choices = [DeltaChoice(delta=DeltaContent(content=response), index=0, finish_reason=None)]
                    yield f"data: {Completion(id=request.id, object='chat.completion.chunk', model=request.model, choices=choices).json()}\n\n"
                # Add a final message with a finish reason to indicate that the chat is done
                choices = [DeltaChoice(delta=DeltaEOS(), index=0, finish_reason="stop")]
                yield f"data: {Completion(id=request.id, object='chat.completion.chunk', model=request.model, choices=choices, finish_reason='stop').json()}\n\n"
                # Add a final message to indicate that the chat is done
                yield "data: [DONE]\n\n"

            return StreamingResponse(openai_streaming_generator(), media_type="text/event-stream")

        # Perform chat completion (batch)
        else:
            content = "".join(
                list(
                    model.chat(
                        messages=messages,
                        max_new_tokens=request.max_tokens,
                        _stream=True,
                    )
                )
            )
            choices = [Choice(message=Message(role="assistant", content=content), finish_reason="stop")]
            return Completion(id=request.id, model=request.model, choices=choices)

    @app.post(f"/{version}/infer", status_code=status.HTTP_201_CREATED)
    def infer(
        request: InferenceRequest,
        client: Client = Depends(get_client),
    ) -> JSONResponse:
        """Perform inference on the given input data.

        $ curl -X "POST" \
            "http://localhost:8000/v1/infer" \
            -H "Content-Type: appication/json" \
            -d '{
                "model_id": "yolox/small",
                "inputs": {
                    "images": ["data:image/jpeg;base64,..."],
                }
            }'

        $ curl -X "POST" \
            "http://localhost:8000/v1/infer" \
            -H "Content-Type: application/json" \
            -d '{
                "model_id": "openai/clip",
                "method": "encode_text",
                "inputs": {
                    "texts": ["fox jumped over the moon"]
                }
            }'

        $ curl -X "POST" \
            "http://localhost:8000/v1/infer" \
            -H "Content-Type: application/json" \
            -d '{
                "model_id": "noop/process",
                "method": "stream_texts",
                "stream": true,
                "inputs": {
                    "texts": ["fox jumped over the moon"]
                }
            }'
        """
        logger.debug(
            f"Decoding input dictionary [model={request.model_id}, method={request.method}, stream={request.stream}]"
        )
        request.inputs = decode_item(request.inputs)
        return _infer(request, client)

    @app.post(f"/{version}/infer/file", status_code=status.HTTP_201_CREATED)
    def infer_file(
        model_id: str = Form(...),
        method: Optional[str] = Form(None),
        file: Optional[UploadFile] = File(None),
        url: Optional[str] = Form(None),
        client: Client = Depends(get_client),
    ) -> JSONResponse:
        """Perform inference on the given input data using multipart/form-data.

        $ curl -X POST \
            'http://localhost:8000/v1/infer/file \
            -H 'accept: application/json' \
            -H 'Content-Type: multipart/form-data' \
            -F 'model_id=yolox/small' \
            -F 'file=@test.jpg'

        $ curl -X POST \
            'http://localhost:8000/v1/infer/file \
            -H 'accept: application/json' \
            -H 'Content-Type: multipart/form-data' \
            -F 'model_id=yolox/small' \
            -F 'url=https://.../test.jpg'

        """
        logger.debug(f"Received inference request [model={model_id}, method={method}]")
        request = InferenceRequest(model_id=model_id, method=method, inputs={})
        file_object: SpooledTemporaryFile = None
        file_basename: str = None
        if file is not None:
            assert isinstance(file.file, SpooledTemporaryFile), f"Invalid file [file={file}]"
            file_object: SpooledTemporaryFile = file.file
            file_basename = Path(file.filename).name
            logger.debug(
                f"Received file upload [model={model_id}, method={method}, filename={file.filename}, basename={file_basename}]"
            )
        elif url is not None:
            assert isinstance(url, str) and url.startswith("http"), f"Invalid url [url={url}]"
            file_object = SpooledTemporaryFile()
            file_object.write(requests.get(url).content)
            file_basename = url.split("?")[0].split("/")[-1]
            logger.debug(f"Downloaded file from url [url={url}, basename={file_basename}]")
        else:
            logger.exception(f"Invalid input data [model={model_id}, method={method}]")
            return JSONResponse(
                content={
                    "error": f"Invalid input data, provide atleast (file, url or json) [model={model_id}, method={method}]"
                },
                status_code=status.HTTP_400_BAD_REQUEST,
            )

        # Check if the file-type is supported
        if not (is_video(file_basename) or is_audio(file_basename) or is_image(file_basename)):
            logger.exception(f"Unsupported file type [filename={file_basename}]")
            return JSONResponse(
                content={"error": f"Unsupported file type [filename={file_basename}]"},
                status_code=status.HTTP_400_BAD_REQUEST,
            )

        # Save the uploaded file to a temporary file and process it
        with as_path(file_object, suffix=Path(file_basename).suffix) as path:
            logger.debug(
                f"Decoding input file [model={model_id}, method={method}, path={path}, size_mb={path.stat().st_size / 1024 / 1024} MB]"
            )
            if path.stat().st_size == 0:
                logger.exception(f"Empty file [path={path}]")
                return JSONResponse(
                    content={"error": f"Empty file [path={path}]"},
                    status_code=status.HTTP_400_BAD_REQUEST,
                )
            # Create the inference request
            ctx = None
            logger.debug(f"Creating inference request [model={model_id}, method={method}, path={path}]")
            if path is not None:
                if is_image(path.name):
                    request.inputs["images"] = [Image.open(str(path))]
                elif is_audio(path.name):
                    ctx = client.UploadFile(path)
                    remote_path = ctx.__enter__()
                    request.inputs["path"] = remote_path
                else:
                    logger.exception(f"Unsupported file type [path={path}]")
                    return JSONResponse(
                        content={"error": f"Unsupported file type [path={path}]"},
                        status_code=status.HTTP_400_BAD_REQUEST,
                    )
            # Perform inference
            return _infer(request, client)

    def _infer(request: InferenceRequest, client: Client) -> JSONResponse:
        """Perform inference on the given input data.

        Args:
            request: Inference request.
            file: Uploaded image / video / audio file for inference.
            client: Inference client.

        Returns:
            Inference response.
        """
        st = time.perf_counter()

        # Check if the model is supported
        logger.debug(f"Initializing module for inference [model={request.model_id}, method={request.method}]")
        try:
            model = client.Module(request.model_id)
        except Exception:
            logger.exception(f"Model '{request.model_id}' not supported")
            return JSONResponse(
                content={"error": f"Model '{request.model_id}' not supported"},
                status_code=status.HTTP_400_BAD_REQUEST,
            )
        logger.debug(f"Initialized module for inference [model={request.model_id}]")

        # Check if the model supports the given method (if provided)
        if request.method is not None and not hasattr(model, request.method):
            logger.exception(f"Method '{request.method}' not supported")
            return JSONResponse(
                content={"error": f"Method '{request.method}' not supported"},
                status_code=status.HTTP_400_BAD_REQUEST,
            )
        logger.debug(f"Initialized module method for inference [model={request.model_id}, method={request.method}]")

        # Handle file uploads as inputs
        inputs = request.inputs

        # Perform inference
        logger.debug(f"Inference [model={request.model_id}, keys={inputs.keys()}]")
        response = model(**inputs, _method=request.method, _stream=request.stream)
        logger.debug(
            f"Inference [model={request.model_id}, , method={request.method}, stream={request.stream}, elapsed={(time.perf_counter() - st) * 1e3:.1f}ms]"
        )

        # Handle response types
        if request.stream:

            def streaming_gen():
                yield from response

            return StreamingResponse(streaming_gen(), media_type="text/event-stream")
        else:
            try:
                return JSONResponse(content=encode_item(response), status_code=status.HTTP_201_CREATED)
            except Exception as e:
                logger.exception(f"Failed to encode response [type={type(response)}, e={e}]")
                raise HTTPException(status_code=500, detail="Image generation failed")

    return app


def main():
    """Main entrypoint for the NOS REST API service / gateway."""
    import argparse

    import uvicorn

    from nos.constants import DEFAULT_HTTP_HOST, DEFAULT_HTTP_PORT
    from nos.logging import logger

    parser = argparse.ArgumentParser(description="NOS REST API Service")
    parser.add_argument("--host", type=str, default=DEFAULT_HTTP_HOST, help="Host address")
    parser.add_argument("--port", type=int, default=DEFAULT_HTTP_PORT, help="Port number")
    parser.add_argument("--workers", type=int, default=1, help="Number of workers")
    parser.add_argument(
        "--reload", action="store_true", help="Reload the REST API service when the source code changes"
    )
    parser.add_argument(
        "--reload-dir",
        type=str,
        default=".",
        help="Directory to watch for changes when reloading the REST API service",
    )
    parser.add_argument("--log-level", type=str, default="info", help="Logging level")
    args = parser.parse_args()
    logger.debug(f"args={args}")

    # Start the NOS REST API service
    logger.debug(
        f"Starting NOS REST API service (host={args.host}, port={args.port}, workers={args.workers}, env={HTTP_ENV})"
    )
    uvicorn.run(
        "nos.server.http._service:app_factory",
        host=args.host,
        port=args.port,
        log_level=args.log_level,
        reload=args.reload,
        reload_dirs=[args.reload_dir],
        workers=args.workers,
        factory=True,
    )


if __name__ == "__main__":
    main()
