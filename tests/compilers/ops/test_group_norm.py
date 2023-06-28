import torch
import torch.nn as nn


def test_group_norm_plugin():
    import torch_tensorrt as torchtrt

    class Norm(torch.nn.Module):
        def __init__(self, C: int):
            super(Norm, self).__init__()
            self.gn = nn.GroupNorm(C // 2, C)

        def forward(self, x):
            return self.gn(x)

    C = 6
    shape = [1, C, 64, 64]
    model = Norm(C).eval().cuda()
    input = torch.randn(shape).to("cuda")

    compile_spec = {
        "inputs": [torchtrt.Input(input.shape, dtype=torch.float, format=torch.contiguous_format)],
        "device": torchtrt.Device("cuda:0"),
        "enabled_precisions": {torch.float},
        "ir": "dyanmo_compile",
    }

    trt_mod = torchtrt.compile(model, **compile_spec)
    print(trt_mod(input))
    # cos_sim = cosine_similarity(model(input), trt_mod(input))
    # assertions.assertTrue(
    #     cos_sim > COSINE_THRESHOLD,
    #     msg=f"Resnet18 TRT outputs don't match with the original model. Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}",
    # )

    # Clean up model env
    torch._dynamo.reset()

    with torch.no_grad():
        torch.cuda.empty_cache()
