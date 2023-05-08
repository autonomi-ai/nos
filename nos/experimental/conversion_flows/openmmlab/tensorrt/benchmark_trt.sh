python mmdeploy/tools/test.py \
    mmdeploy/configs/mmdet/detection/detection_tensorrt_dynamic-320x320-1344x1344.py \
    mmdetection/configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py \
    --model converted_trt/end2end.engine \
    --device cuda \
    --speed-test