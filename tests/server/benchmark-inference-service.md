### GPU benchmarks (0.1.0a0 - 8/15/2023)

```bash
Timing Records (0.1.0a0) - OMP_NUM_THREADS=32 - GPU
                                                 desc    b      n  elapsed         shape image_type backend  latency_ms      fps      date  version
0                     noop/process-images_1x224x224x3    1   2227      5.0    (224, 224)    ndarray     cpu        2.25    445.4  20231011  0.1.0a0
1                    noop/process-images_16x224x224x3   16  21536      5.0    (224, 224)    ndarray     cpu        0.23   4307.2  20231011  0.1.0a0
2                   noop/process-images_256x224x224x3  256  52480      5.0    (224, 224)    ndarray     cpu        0.10  10496.0  20231011  0.1.0a0
3                     noop/process-images_1x640x480x3    1   1858      5.0    (640, 480)    ndarray     cpu        2.69    371.6  20231011  0.1.0a0
4                    noop/process-images_16x640x480x3   16  10528      5.0    (640, 480)    ndarray     cpu        0.47   2105.6  20231011  0.1.0a0
5                   noop/process-images_256x640x480x3  256  11776      5.0    (640, 480)    ndarray     cpu        0.42   2355.2  20231011  0.1.0a0
6                    noop/process-images_1x1280x720x3    1   1291      5.0   (1280, 720)    ndarray     cpu        3.87    258.2  20231011  0.1.0a0
7                   noop/process-images_16x1280x720x3   16   3056      5.0   (1280, 720)    ndarray     cpu        1.64    611.2  20231011  0.1.0a0
8                   noop/process-images_1x2880x1620x3    1    648      5.0  (2880, 1620)    ndarray     cpu        7.72    129.6  20231011  0.1.0a0
9                  noop/process-images_16x2880x1620x3   16    816      5.0  (2880, 1620)    ndarray     cpu        6.13    163.2  20231011  0.1.0a0
10                    noop/process-images_1x224x224x3    1   2174      5.0    (224, 224)      Image     cpu        2.30    434.8  20231011  0.1.0a0
11                   noop/process-images_16x224x224x3   16  17328      5.0    (224, 224)      Image     cpu        0.29   3465.6  20231011  0.1.0a0
12                  noop/process-images_256x224x224x3  256  29184      5.0    (224, 224)      Image     cpu        0.17   5836.8  20231011  0.1.0a0
13                    noop/process-images_1x640x480x3    1   1690      5.0    (640, 480)      Image     cpu        2.96    338.0  20231011  0.1.0a0
14                   noop/process-images_16x640x480x3   16   5200      5.0    (640, 480)      Image     cpu        0.96   1040.0  20231011  0.1.0a0
15                  noop/process-images_256x640x480x3  256   4352      5.0    (640, 480)      Image     cpu        1.15    870.4  20231011  0.1.0a0
16                   noop/process-images_1x1280x720x3    1   1010      5.0   (1280, 720)      Image     cpu        4.95    202.0  20231011  0.1.0a0
17                  noop/process-images_16x1280x720x3   16   1616      5.0   (1280, 720)      Image     cpu        3.09    323.2  20231011  0.1.0a0
18                  noop/process-images_1x2880x1620x3    1    350      5.0  (2880, 1620)      Image     cpu       14.29     70.0  20231011  0.1.0a0
19                 noop/process-images_16x2880x1620x3   16    272      5.0  (2880, 1620)      Image     cpu       18.38     54.4  20231011  0.1.0a0
20           openai/clip-vit-base-patch32_1x224x224x3    1    352      5.0    (224, 224)    ndarray     cpu       14.20     70.4  20231011  0.1.0a0
21          openai/clip-vit-base-patch32_16x224x224x3   16   1056      5.0    (224, 224)    ndarray     cpu        4.73    211.2  20231011  0.1.0a0
22         openai/clip-vit-base-patch32_256x224x224x3  256    768      5.0    (224, 224)    ndarray     cpu        6.51    153.6  20231011  0.1.0a0
23           openai/clip-vit-base-patch32_1x640x480x3    1    278      5.0    (640, 480)    ndarray     cpu       17.99     55.6  20231011  0.1.0a0
24          openai/clip-vit-base-patch32_16x640x480x3   16    592      5.0    (640, 480)    ndarray     cpu        8.45    118.4  20231011  0.1.0a0
25         openai/clip-vit-base-patch32_256x640x480x3  256    256      5.0    (640, 480)    ndarray     cpu       19.53     51.2  20231011  0.1.0a0
26           openai/clip-vit-base-patch32_1x224x224x3    1    361      5.0    (224, 224)      Image     cpu       13.85     72.2  20231011  0.1.0a0
27          openai/clip-vit-base-patch32_16x224x224x3   16   1040      5.0    (224, 224)      Image     cpu        4.81    208.0  20231011  0.1.0a0
28         openai/clip-vit-base-patch32_256x224x224x3  256    768      5.0    (224, 224)      Image     cpu        6.51    153.6  20231011  0.1.0a0
29           openai/clip-vit-base-patch32_1x640x480x3    1    272      5.0    (640, 480)      Image     cpu       18.38     54.4  20231011  0.1.0a0
30          openai/clip-vit-base-patch32_16x640x480x3   16    560      5.0    (640, 480)      Image     cpu        8.93    112.0  20231011  0.1.0a0
31         openai/clip-vit-base-patch32_256x640x480x3  256    512      5.0    (640, 480)      Image     cpu        9.77    102.4  20231011  0.1.0a0
32                           yolox/medium_1x640x480x3    1    280      5.0    (640, 480)    ndarray     cpu       17.86     56.0  20231011  0.1.0a0
33                          yolox/medium_16x640x480x3   16    528      5.0    (640, 480)    ndarray     cpu        9.47    105.6  20231011  0.1.0a0
34                         yolox/medium_256x640x480x3  256    256      5.0    (640, 480)    ndarray     cpu       19.53     51.2  20231011  0.1.0a0
35                          yolox/medium_1x1280x720x3    1    187      5.0   (1280, 720)    ndarray     cpu       26.74     37.4  20231011  0.1.0a0
36                         yolox/medium_16x1280x720x3   16    144      5.0   (1280, 720)    ndarray     cpu       34.72     28.8  20231011  0.1.0a0
37                         yolox/medium_1x2880x1620x3    1     24      5.0  (2880, 1620)    ndarray     cpu      208.33      4.8  20231011  0.1.0a0
38                        yolox/medium_16x2880x1620x3   16     16      5.0  (2880, 1620)    ndarray     cpu      312.50      3.2  20231011  0.1.0a0
39                           yolox/medium_1x640x480x3    1    277      5.0    (640, 480)      Image     cpu       18.05     55.4  20231011  0.1.0a0
40                          yolox/medium_16x640x480x3   16    480      5.0    (640, 480)      Image     cpu       10.42     96.0  20231011  0.1.0a0
41                         yolox/medium_256x640x480x3  256    256      5.0    (640, 480)      Image     cpu       19.53     51.2  20231011  0.1.0a0
42                          yolox/medium_1x1280x720x3    1    177      5.0   (1280, 720)      Image     cpu       28.25     35.4  20231011  0.1.0a0
43                         yolox/medium_16x1280x720x3   16    144      5.0   (1280, 720)      Image     cpu       34.72     28.8  20231011  0.1.0a0
44                         yolox/medium_1x2880x1620x3    1     25      5.0  (2880, 1620)      Image     cpu      200.00      5.0  20231011  0.1.0a0
45                        yolox/medium_16x2880x1620x3   16     16      5.0  (2880, 1620)      Image     cpu      312.50      3.2  20231011  0.1.0a0
46  torchvision/fasterrcnn-mobilenet-v3-large-320-...    1    250      5.0    (640, 480)    ndarray     cpu       20.00     50.0  20231011  0.1.0a0
47  torchvision/fasterrcnn-mobilenet-v3-large-320-...   16    672      5.0    (640, 480)    ndarray     cpu        7.44    134.4  20231011  0.1.0a0
48  torchvision/fasterrcnn-mobilenet-v3-large-320-...  256    768      5.0    (640, 480)    ndarray     cpu        6.51    153.6  20231011  0.1.0a0
49  torchvision/fasterrcnn-mobilenet-v3-large-320-...    1    187      5.0   (1280, 720)    ndarray     cpu       26.74     37.4  20231011  0.1.0a0
50  torchvision/fasterrcnn-mobilenet-v3-large-320-...   16    336      5.0   (1280, 720)    ndarray     cpu       14.88     67.2  20231011  0.1.0a0
51  torchvision/fasterrcnn-mobilenet-v3-large-320-...    1     62      5.0  (2880, 1620)    ndarray     cpu       80.65     12.4  20231011  0.1.0a0
52  torchvision/fasterrcnn-mobilenet-v3-large-320-...   16     48      5.0  (2880, 1620)    ndarray     cpu      104.17      9.6  20231011  0.1.0a0
53  torchvision/fasterrcnn-mobilenet-v3-large-320-...    1    249      5.0    (640, 480)      Image     cpu       20.08     49.8  20231011  0.1.0a0
54  torchvision/fasterrcnn-mobilenet-v3-large-320-...   16    864      5.0    (640, 480)      Image     cpu        5.79    172.8  20231011  0.1.0a0
55  torchvision/fasterrcnn-mobilenet-v3-large-320-...  256    512      5.0    (640, 480)      Image     cpu        9.77    102.4  20231011  0.1.0a0
56  torchvision/fasterrcnn-mobilenet-v3-large-320-...    1    188      5.0   (1280, 720)      Image     cpu       26.60     37.6  20231011  0.1.0a0
57  torchvision/fasterrcnn-mobilenet-v3-large-320-...   16    288      5.0   (1280, 720)      Image     cpu       17.36     57.6  20231011  0.1.0a0
58  torchvision/fasterrcnn-mobilenet-v3-large-320-...    1     55      5.0  (2880, 1620)      Image     cpu       90.91     11.0  20231011  0.1.0a0
59  torchvision/fasterrcnn-mobilenet-v3-large-320-...   16     48      5.0  (2880, 1620)      Image     cpu      104.17      9.6  20231011  0.1.0a0
2023-10-11 16:42:07.570 | INFO     | test_inference_service:test_benchmark_inference_service_noop:281 - Saved timing records to /home/spillai/autonomi/nos/.data/benchmark/nos-cpu-in
ference-benchmark--0-1-0a0--20231011.json
```
