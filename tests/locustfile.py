from locust import HttpUser, task
from PIL import Image

from nos.server.http._utils import encode_dict

NOS_TEST_IMAGE = "/home/scott/dev/nos/tests/test_data/test.jpg"

class NosRestUser(HttpUser):
    @task
    def embed(self):

        data = {
            "task": "image_embedding",
            "model_name": "openai/clip",
            "inputs": {
                "images": Image.open(NOS_TEST_IMAGE),
            },
        }

        response = self.client.post(
            "/infer",
            headers={"Content-Type": "application/json"},
            json=encode_dict(data),
        )

        print('Status code:', response.status_code)
