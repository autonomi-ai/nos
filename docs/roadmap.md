# Roadmap

## Hardware Support
We currently plan to support the following hardware:

- [x] GPUs
    * [x] NVIDIA (A100, A10G, T4, 4090, 2080)
    * [ ] AMD GPUs (MI250)
- [x] Cloud Providers
    * [x] AWS (g4/g5dn/p3/p4dn)
        - [ ] AWS Inferentia inf1/inf2
        - [ ] Intel Habana Gaudi
    * [ ] GCP (g2/a1/n1)
        - [ ] Google TPUv3
---
## Model Hub
 - **Text-to-Image**
    * [x] [Stable Diffusion v1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5)
    * [x] [Stable Diffusion v2.0](https://huggingface.co/stabilityai/stable-diffusion-2)
 - **Image/Text-to-Vec**
    * [x] [OpenAI CLIP ViT](https://huggingface.co/openai/clip-vit-base-patch32)
    * [x] [Laion CLIP ViT](https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K)
 - **Object Detection**
    * [x] [FasterRCNN](https://pytorch.org/vision/main/models/faster_rcnn.html)
    * [ ] [EfficientDet (OpenMMLab)](https://github.com/open-mmlab/mmdetection/tree/main/projects/EfficientDet)
    * [ ] [FasterRCNN (OpenMMLab)](https://github.com/open-mmlab/mmdetection/blob/main/mmdet/models/detectors/faster_rcnn.py)
---
### Model Hub: Coming Soon

 - **Audio-to-Text**
    * [ ] [OpenAI Whisper](https://huggingface.co/openai/whisper-large-v2)
    * [ ] [WhisperX](https://github.com/m-bain/whisperX)
 - **Object Tracking**
    * [ ] [MMtracking (OpenMMLab)](https://github.com/open-mmlab/mmtracking)
---
## Model Optimizations
 - [**NVIDIA TensorRT**](https://developer.nvidia.com/tensorrt)
 - [**Facebook AITemplate**](https://github.com/facebookincubator/AITemplate)
