import dataclasses
import time
from dataclasses import field
from typing import Any, Dict

from fastapi import Depends, FastAPI, status
from fastapi.responses import JSONResponse
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

from nos.client import DEFAULT_GRPC_PORT, Client
from nos.logging import logger
from nos.protoc import import_module
from nos.version import __version__

from ._utils import decode_dict, encode_dict


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


@dataclasses.dataclass
class InferenceService:
    """HTTP server application for NOS API."""

    version: str = field(default="v1")
    """NOS version."""

    grpc_port: int = field(default=DEFAULT_GRPC_PORT)
    """gRPC port number."""

    debug: bool = field(default=False)
    """Debug mode."""

    app: FastAPI = field(init=False, default=None)
    """FastAPI app."""

    client: Client = field(init=False, default=None)
    """Inference client."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    """Model configuration."""

    def __post_init__(self):
        """Post initialization."""
        self.app = FastAPI(
            title="NOS REST API",
            description=f"NOS REST API (version={__version__}, api_version={self.version})",
            version=self.version,
            debug=True,
        )
        logger.debug(f"Starting NOS REST API (version={__version__})")

        self.client = Client(f"[::]:{self.grpc_port}")
        logger.debug(f"Connecting to gRPC server (address={self.client.address})")

        self.client.WaitForServer(timeout=60)
        runtime = self.client.GetServiceRuntime()
        version = self.client.GetServiceVersion()
        logger.debug(f"Connected to gRPC server (address={self.client.address}, runtime={runtime}, version={version})")


def app(version: str = "v1", grpc_port: int = DEFAULT_GRPC_PORT, debug: bool = False) -> FastAPI:
    nos_app = InferenceService(version=version, grpc_port=grpc_port, debug=debug)
    app = nos_app.app

    def get_client() -> Client:
        """Get the inference client."""
        return nos_app.client

    @app.get("/health", status_code=status.HTTP_200_OK)
    def health(client: Client = Depends(get_client)) -> JSONResponse:
        """Check if the server is alive."""
        return JSONResponse(
            content={"status": "ok" if client.IsHealthy() else "not_ok"}, status_code=status.HTTP_200_OK
        )

    @app.post("/infer", status_code=status.HTTP_201_CREATED)
    def infer(request: InferenceRequest, client: Client = Depends(get_client)) -> JSONResponse:
        """Perform inference on the given input data.

        $ curl -X "POST" \
            "http://localhost:8000/infer" \
            -H "Content-Type: appication/json" \
            -d '{
                "model_id": "yolox/small",
                "inputs": {
                    "images": ["data:image/jpeg;base64,..."],
                }
            }'

        $ curl -X "POST" \
            "http://localhost:8000/infer" \
            -H "Content-Type: application/json" \
            -d '{
                "model_id": "openai/clip",
                "method": "encode_text",
                "inputs": {
                    "texts": ["fox jumped over the moon"]
                }
            }' | jq

        Args:
            request: Inference request.
            file_object: Uploaded image / video / audio file for inference.
            client: Inference client.

        Returns:
            Inference response.
        """
        st = time.perf_counter()

        logger.debug(f"Initializing module for inference [model={request.model_id}, method={request.method}]")
        try:
            model = client.Module(request.model_id)
        except Exception:
            logger.error(f"Model '{request.model_id}' not supported")
            return JSONResponse(
                content={"error": f"Model '{request.model_id}' not supported"},
                status_code=status.HTTP_400_BAD_REQUEST,
            )
        logger.debug(f"Initialized module for inference [model={request.model_id}, method={request.method}]")

        logger.debug(f"Decoding input dictionary [model={request.model_id}, method={request.method}]")
        inputs = decode_dict(request.inputs)
        logger.debug(f"Decoded input dictionary [model={request.model_id}, method={request.method}]")
        logger.debug(f"Inference [model={request.model_id}, keys={inputs.keys()}]")
        response = model(**inputs, _method=request.method)
        logger.debug(
            f"Inference [model={request.model_id}, , method={request.method}, elapsed={(time.perf_counter() - st) * 1e3:.1f}ms]"
        )
        return JSONResponse(content=encode_dict(response), status_code=status.HTTP_201_CREATED)

    return app


def main():
    """Main entrypoint for the NOS REST API service.

    We start the NOS gRPC server as part of this entrypoint to ensure that the gRPC client proxy
    is able to connect to the gRPC server before starting the REST API service.
    """
    import argparse
    import os

    import uvicorn

    import nos
    from nos.client import Client
    from nos.constants import DEFAULT_GRPC_PORT, DEFAULT_HTTP_PORT, NOS_HTTP_MAX_WORKER_THREADS
    from nos.logging import logger

    parser = argparse.ArgumentParser(description="NOS REST API Service")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address")
    parser.add_argument("--port", type=int, default=DEFAULT_HTTP_PORT, help="Port number")
    parser.add_argument("--workers", type=int, default=NOS_HTTP_MAX_WORKER_THREADS, help="Number of workers")
    parser.add_argument(
        "--server", type=bool, default=False, help="Initialize the gRPC server with the REST API service"
    )
    args = parser.parse_args()
    logger.debug(f"args={args}")

    if args.server:
        # Start the NOS gRPC Server
        logger.debug(f"Starting NOS gRPC server (port={DEFAULT_GRPC_PORT})")
        nos.init(runtime="auto", logging_level=os.environ.get("NOS_LOGGING_LEVEL", "INFO"))

    # Wait for the gRPC server to be ready
    logger.debug(f"Initializing gRPC client (port={DEFAULT_GRPC_PORT})")
    client = Client(f"[::]:{DEFAULT_GRPC_PORT}")
    logger.debug(f"Initialized gRPC client, connecting to gRPC server (address={client.address})")
    if not client.WaitForServer(timeout=180, retry_interval=5):
        raise RuntimeError("Failed to connect to gRPC server")
    if not client.IsHealthy():
        raise RuntimeError("gRPC server is not healthy")

    # Start the NOS REST API service
    logger.debug(f"Starting NOS REST API service (host={args.host}, port={args.port}, workers={args.workers})")
    uvicorn.run(
        "nos.server.http._service:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        log_level="info",
        factory=True,
    )


if __name__ == "__main__":
    main()
