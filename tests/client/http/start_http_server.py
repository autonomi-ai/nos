# Create a test app to run load testing against:
import uvicorn
from nos.server.http._service import app
import nos
from nos.constants import DEFAULT_GRPC_PORT

# Init a nos backend for REST API:
nos.init(runtime='gpu', logging_level='DEBUG')

client = nos.InferenceClient()

client.WaitForServer()
assert client.IsHealthy()

# Start the Fast API app with uvicorn:
uvicorn.run(app(), host="localhost", port=8000, workers=1, log_level="info")
