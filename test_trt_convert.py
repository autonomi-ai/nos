import onnx
from torchvision.models import resnet50
import torch
import tensorrt as trt
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit

dummy_input = torch.randn(1, 3, 224, 224)
model = resnet50(pretrained=True)

# Convert to onnx
onnx_path = "resnet50.onnx"
torch.onnx.export(model, dummy_input, onnx_path, verbose=True)

# Onnx provides its own trt backend. This is difficult to install, 
# onnx-tensorrt needs to be build from source into its own container.

# Convert to trt using trt
trt_logger = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(trt_logger)
trt_network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(trt_network, trt_logger)
res = parser.parse_from_file(onnx_path)

# generate the engine file
config = builder.create_builder_config()
serialized_engine = builder.build_serialized_network(trt_network, config)
engine_path = "resnet50.engine" 
with open(engine_path, "wb") as f:
    f.write(serialized_engine)

runtime = trt.Runtime(trt_logger)
with open(engine_path, "rb") as f:
    engine = f.read()
    deserialized_engine = runtime.deserialize_cuda_engine(engine)

print("Create execution context")
execution_context = deserialized_engine.create_execution_context()
stream = cuda.Stream()

# Setup input allocations for trt engine
input_shape = (1, 3, 224, 224)
output_shape = (1, 1000)

print("Create input allocations...")
input_host = cuda.pagelocked_empty(trt.volume(input_shape), dtype=np.float32)
output_host = cuda.pagelocked_empty(trt.volume(output_shape), dtype=np.float32)
# Allocate device memory for inputs and outputs.
input_device = cuda.mem_alloc(input_host.nbytes)
output_device = cuda.mem_alloc(output_host.nbytes)

# Run trt inference with the bound inputs and outputs
print("Run inference...")
cuda.memcpy_htod_async(input_device, input_host, stream)
print("output host: ", output_host)
execution_context.execute_async_v2([int(input_device), int(output_device)], stream.handle, None)
cuda.memcpy_dtoh_async(output_host, output_device, stream)
stream.synchronize()
print("output host: ", output_host)


