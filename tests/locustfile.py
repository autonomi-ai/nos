from locust import HttpUser, task
from PIL import Image

from nos.server.http._utils import encode_dict
from nos.test.utils import NOS_TEST_IMAGE


data = {
    "model_id": "noop/process-images",
    "inputs": {
        "images": Image.open(NOS_TEST_IMAGE),
    },
}
encoded_data = encode_dict(data)


class InferenceServiceUser(HttpUser):
    @task
    def embed(self):
        self.client.post(
            "/infer",
            headers={"Content-Type": "application/json"},
            json=encoded_data,
        )
