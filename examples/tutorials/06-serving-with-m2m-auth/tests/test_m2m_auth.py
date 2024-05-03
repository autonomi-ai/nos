import requests


if __name__ == "__main__":
    BASE_URL = "http://localhost:8000"

    # Test health
    response = requests.get(f"{BASE_URL}/v1/health")
    response.raise_for_status()

    # Test model info without authentication
    response = requests.get(f"{BASE_URL}/v1/models")
    assert response.status_code == 401, "Expected 401 Unauthorized"

    # Test model info with invalid authentication
    response = requests.get(f"{BASE_URL}/v1/models", headers={"X-Api-Key": "invalid-api-key"})
    assert response.status_code == 403, "Expected 403 Forbidden"

    # Test model info with valid authentication
    response = requests.get(f"{BASE_URL}/v1/models", headers={"X-Api-Key": "sk-test-key-1"})
    response.raise_for_status()
    assert response.status_code == 200, "Expected 200 OK"

    # Test model inference without authentication
    response = requests.get(f"{BASE_URL}/v1/models", headers={"X-Api-Key": "sk-test-key-2"})
    response.raise_for_status()
    assert response.status_code == 200, "Expected 200 OK"
