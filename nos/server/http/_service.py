from dataclasses import dataclass, field

from fastapi import Depends, FastAPI, status
from fastapi.responses import JSONResponse

from nos.client import DEFAULT_GRPC_PORT, Client
from nos.logging import logger
from nos.protoc import import_module
from nos.version import __version__

from ._types import InferenceRequest
from ._utils import decode_dict, encode_dict


nos_service_pb2 = import_module("nos_service_pb2")
nos_service_pb2_grpc = import_module("nos_service_pb2_grpc")


@dataclass
class InferenceService:
    """HTTP server application for NOS API."""

    version: str = "v1"
    """NOS version."""

    grpc_port: int = DEFAULT_GRPC_PORT
    """gRPC port number."""

    debug: bool = False
    """Debug mode."""

    app: FastAPI = field(init=False, default=None)
    """FastAPI app."""

    client: Client = field(init=False, default=None)
    """Inference client."""

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
        "http://localhost:8000/v1/infer" \
        -H "Content-Type: appication/json" \
        -d '{
            "model_id": "yolox/small",
            "inputs": {
                "images": ["data:image/jpeg;base64,..."],
            }
        }'

        Args:
            request: Inference request.
            file_object: Uploaded image / video / audio file for inference.
            client: Inference client.

        Returns:
            Inference response.
        """
        try:
            model = client.Module(request.model_id)
        except Exception:
            logger.error(f"Model '{request.model_id}' not supported")
            return JSONResponse(
                content={"error": f"Model '{request.model_id}' not supported"},
                status_code=status.HTTP_400_BAD_REQUEST,
            )

        inputs = decode_dict(request.inputs)
        logger.debug(f"Decoded json dictionary [inputs={inputs}]")
        logger.debug(f"Inference [model={request.model_id}, keys={inputs.keys()}]")
        response = model(**inputs)
        logger.debug(f"Inference [model={request.model_id}, response={response}]")
        return JSONResponse(content=encode_dict(response), status_code=status.HTTP_201_CREATED)

    return app


def main():
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description="NOS REST API Service")
    parser.add_argument("--host", type=str, default="localhost", help="Host address")
    parser.add_argument("--port", type=int, default=8000, help="Port number")
    parser.add_argument("--workers", type=int, default=4, help="Number of workers")
    args = parser.parse_args()

    uvicorn.run(app(), host=args.host, port=args.port, workers=args.workers, log_level="info")


if __name__ == "__main__":
    main()
