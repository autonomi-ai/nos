import torch
from PIL import Image
import tensorrt as trt
import torchvision.transforms.functional as F
import onnx
from torchvision.models import resnet50

test_img = Image.open("/home/scott/dev/nos/tests/test_data/test.jpg").resize((640,480))
images = torch.stack([F.to_tensor(test_img)])
cuda_device = "cuda"
images = images.to(cuda_device).contiguous()

trt.init_libnvinfer_plugins(trt.Logger(), '')
print("loaded plugins")

engine_file_path = "/home/scott/dev/nos/nos/models/openmmlab/mmdetection/end2end.engine"
with open(engine_file_path, "rb") as engine_file, trt.Runtime(trt.Logger()) as runtime:
            print("try to deserialize")
            engine = runtime.deserialize_cuda_engine(engine_file.read())
            print("try to create context")
            # context = engine.create_execution_context()
print(engine.bindings)