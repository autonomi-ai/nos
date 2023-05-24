import logging
from io import BytesIO

from fastapi import FastAPI, HTTPException, status
from ray import serve
from ray.serve.handle import RayServeDeploymentHandle
from rich.console import Console

from nos.hub import TaskType
from nos.serve.service import (
    Image2VecRequest,
    ImageResponse,
    PredictionRequest,
    PredictionResponse,
    Text2ImageRequest,
    Text2VecRequest,
    VecResponse,
)


logger = logging.getLogger(__name__)
app = FastAPI()
console = Console()

# Create the API ingress
# TODO (spillai): Generalize this to support multiple ingress types
@serve.deployment(num_replicas=1, route_prefix="/")
@serve.ingress(app)
class APIIngress:
    def __init__(self, handle: RayServeDeploymentHandle, predict: str = "__call__") -> None:
        self.handle = handle
        self.predict = predict

    @app.get("/health", status_code=status.HTTP_200_OK)
    async def _health(self):
        return {"status": "ok"}

    @app.post("/predict", response_model=PredictionResponse, status_code=status.HTTP_201_CREATED)
    async def predict(self, request: PredictionRequest):
        """Predict using the model."""
        logger.error(f"Received request: {request}")
        if request.method == TaskType.IMAGE_GENERATION.value:
            req: Text2ImageRequest = request.request
            if not len(req.prompt):
                raise HTTPException(status_code=400, detail="Prompt cannot be empty.")
            logger.info(f"Generating image with prompt: {req.prompt}")
            response_ref = await self.handle.__call__.remote(req.prompt, height=req.height, width=req.width)
            (image,) = await response_ref
            file_stream = BytesIO()
            image.save(file_stream, "PNG")
            return PredictionResponse(response=ImageResponse.from_pil(image))

        elif request.method == TaskType.TEXT_EMBEDDING.value:
            req: Text2VecRequest = request.request
            logger.info(f"Encoding text: {req.text}")
            response_ref = await self.handle.encode_text.remote(req.text)
            embedding = await response_ref
            return PredictionResponse(response=VecResponse.from_numpy(embedding.ravel()))

        elif request.method == TaskType.IMAGE_EMBEDDING.value:
            req: Image2VecRequest = request.request
            response_ref = await self.handle.encode_image.remote(req.image)
            embedding = await response_ref
            return PredictionResponse(response=VecResponse.from_numpy(embedding.ravel()))

        else:
            raise NotImplementedError(f"Method {request.method} not implemented.")
