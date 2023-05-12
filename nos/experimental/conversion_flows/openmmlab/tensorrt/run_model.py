#!/usr/bin/env python3

import argparse
import os

from mmdeploy.apis import inference_model


def main():
    parser = argparse.ArgumentParser(description="Run inference on a single image")
    parser.add_argument("--image_path", type=str, help="path to image", default="test.jpg")
    args = parser.parse_args()
    print(args)

    workspace_path = os.path.join(os.path.expanduser("~"), "workspace")

    model_cfg = os.path.join(workspace_path, "mmdetection/configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py")
    deploy_cfg = os.path.join(
        workspace_path, "mmdeploy/configs/mmdet/detection/detection_tensorrt_dynamic-320x320-1344x1344.py"
    )
    backend_files = [os.path.join(workspace_path, "mmdeploy_model/faster-rcnn/end2end.engine")]
    device = "cuda:0"

    print(f"model_cfg: {model_cfg}")
    print(f"deploy_cfg: {deploy_cfg}")
    print(f"backend_files: {backend_files}")
    print(f"device: {device}")

    inference_model(
        model_cfg,  # defines the model we want to run
        deploy_cfg,  # defines the backend we want to run against (TRT)
        backend_files,  # list of files we need to run the backend (TRT engine)
        args.image_path,
        device,
    )


if __name__ == "__main__":
    main()
