Conversion setup for faster-rcnn and tensorrt
1) docker build . -t mmdeploy-trt-conversion
2) docker run -it --gpus all  mmdeploy-trt-conversion
(optional) mount existing trt engine and checkpoints with -v checkpoints/*:/root/workspace/checkpoints/ -v /faster-rcnn:/root/workspace/faster-rcnn
3) ./convert_faster_rcnn.sh
4) ./run_model.py