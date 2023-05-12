import time
from dataclasses import dataclass

import requests


NOS_SERVE_DEFAULT_HTTP_HOST = "127.0.0.1"
NOS_SERVE_DEFAULT_HTTP_PORT = 6169


@dataclass
class SimpleClient:
    host: str = NOS_SERVE_DEFAULT_HTTP_HOST
    port: int = NOS_SERVE_DEFAULT_HTTP_PORT
    timeout: int = 10

    def is_healthy(self, timeout: int = 10) -> bool:
        """Check if the server is healthy."""
        response = requests.get(f"http://{self.host}:{self.port}/health", timeout=timeout)
        return response.status_code == 200

    def wait(self, timeout: int = 30) -> None:
        """Wait for the server to be healthy."""
        st = time.time()
        while time.time() - st <= timeout:
            try:
                health = self.is_healthy()
                if health:
                    return
            except requests.exceptions.ConnectionError:
                time.sleep(1)
                continue
        raise TimeoutError("Failed to connect.")

    def post(self, route: str, **kwargs) -> requests.Response:
        """Send a POST request to the server."""
        return requests.post(f"http://{self.host}:{self.port}/{route}", **kwargs)
