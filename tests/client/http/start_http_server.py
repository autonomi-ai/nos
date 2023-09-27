# Create a test app to run load testing against:
import uvicorn

from nos.server.http._service import app


uvicorn.run(app(), host="localhost", port=8000, workers=1, log_level="info")
