python mmdeploy/tools/test.py \
    mmdeploy/configs/mmdet/detection/detection_onnxruntime_dynamic.py \
    mmdetection/configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py \
    --model converted_onnx/end2end.onnx \
    --device cuda \
    --speed-test