from locust import HttpUser, task
from PIL import Image

from nos.server.http._utils import encode_dict
from nos.test.utils import NOS_TEST_IMAGE


NOS_TEST_IMAGE = "/home/scott/dev/nos/tests/test_data/test.jpg"

data = {
    "task": "custom",
    "model_name": "noop/process-images",
    "inputs": {
        "images": Image.open(NOS_TEST_IMAGE),
    },
}

encoded_data = encode_dict(data)


class NosRestUser(HttpUser):
    @task
    def embed(self):
        self.client.post(
            "/infer",
            headers={"Content-Type": "application/json"},
            json=encoded_data,
        )
