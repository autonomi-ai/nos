from locust import HttpUser, task
from PIL import Image

from nos.server.http._utils import encode_dict


NOS_TEST_IMAGE = "tests/data/test_image.jpg"


class NosRestUser(HttpUser):
    @task
    def embed(self):
        self.data = {
            "task": "image_embedding",
            "model_name": "openai/clip",
            "inputs": {
                "images": Image.open(NOS_TEST_IMAGE),
            },
        }
        self.client.post(
            "/infer",
            headers={"Content-Type": "application/json"},
            json=encode_dict(self.data),
        )
