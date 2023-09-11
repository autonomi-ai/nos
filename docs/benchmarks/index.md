## Benchmarks
 - [0.0.9a2](./benchmarks.md#009a2)
 - [0.0.7a3](./benchmarks.md#007a3)

## Profiles
 - [NVIDIA GeForce RTX 4090](./profiles.md#nvidia-geforce-rtx-4090)
 - [NVIDIA GeForce RTX 2080 Ti](./profiles.md#nvidia-geforce-rtx-2080-ti)
 - [NVIDIA GeForce RTX 2080](./profiles.md#nvidia-geforce-rtx-2080)

## FLOPs Reference

Below are the raw TFLOPs of the different GPUs available from cloud providers.

| Vendor | Model | Arch   | FP32 | Mixed-precision | FP16 | Source             |
| ------ | ----- | ------ | ---- | --------------- | ---- | ------------------ |
| NVIDIA | A100  | Ampere | 19.5 | 156             | 312  | [Datasheet][a100]  |
| NVIDIA | A10G  | Ampere | 35   | 35              | 70   | [Datasheet][a10g]  |
| NVIDIA | A6000 | Ampere | 38   | ?               | ?    | [Datasheet][a6000] |
| NVIDIA | V100  | Volta  | 14   | 112             | 28   | [Datasheet][v100]  |
| NVIDIA | T4    | Turing | 8.1  | 65              | ?    | [Datasheet][t4]    |
| NVIDIA | P4    | Pascal | 5.5  | N/A             | N/A  | [Datasheet][p4]    |
| NVIDIA | P100  | Pascal | 9.3  | N/A             | 18.7 | [Datasheet][p100]  |
| NVIDIA | K80   | Kepler | 8.73 | N/A             | N/A  | [Datasheet][k80]   |
| NVIDIA | A40   | Ampere | 37   | 150             | 150  | [Datasheet][a40]   |
| AMD    | MI250 | CDNA   | 90.5 | -           | 362  | [Datasheet][mi200] |
| AMD    | MI250X | CDNA   | 95.7 | 184.8           | 383  | [Datasheet][mi200] |
| AWS    | inf1  | Inferentia | 16 | 128           | 256  | [Datasheet][inf1]  |
| AWS    | inf2  | Inferentia | 32 | 256           | 512  | [Datasheet][inf2]  |
| Habana | Gaudi | Goya   | 32   | 256             | 512  | [Datasheet][gaudi] |
| Tenstorrent | Grayskull e300 | Sparsity | 1.6 | 12.8 | 25.6 | [Datasheet][grayskull] |
| Apple  | M1    | - | 8   | 64              | 128  | [Datasheet][m1]    |
| Apple  | M1 Ultra | - | 10 | 80            | 160  | [Datasheet][m1u]   |
| Apple  | M1 Max | - | 12 | 96              | 192  | [Datasheet][m1max] |
| Apple  | M2    | - | 10  | 80              | 160  | [Datasheet][m2]    |
| Apple  | M2 Pro | - | 12  | 96              | 192  | [Datasheet][m2pro] |


[a100]: https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf
[a10g]: https://d1.awsstatic.com/product-marketing/ec2/NVIDIA_AWS_A10G_DataSheet_FINAL_02_17_2022.pdf
[a6000]: https://www.nvidia.com/content/dam/en-zz/Solutions/design-visualization/quadro-product-literature/proviz-print-nvidia-rtx-a6000-datasheet-us-nvidia-1454980-r9-web%20(1).pdf
[v100]: https://images.nvidia.com/content/technologies/volta/pdf/tesla-volta-v100-datasheet-letter-fnl-web.pdf
[t4]: https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-t4/t4-tensor-core-datasheet-951643.pdf
[p4]: https://images.nvidia.com/content/pdf/tesla/184457-Tesla-P4-Datasheet-NV-Final-Letter-Web.pdf
[p100]: https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-p100/pdf/nvidia-tesla-p100-PCIe-datasheet.pdf
[k80]: https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-product-literature/Tesla-K80-BoardSpec-07317-001-v05.pdf
[a40]: https://images.nvidia.com/content/Solutions/data-center/a40/nvidia-a40-datasheet.pdf
[mi200]: https://www.amd.com/system/files/documents/amd-instinct-mi200-datasheet.pdf
[mi210]: https://www.amd.com/system/files/documents/amd-instinct-mi210-brochure.pdf
[inf1]: https://d1.awsstatic.com/events/reinvent/2019/REPEAT_1_Deliver_high_performance_ML_inference_with_AWS_Inferentia_CMP324-R1.pdf
[inf2]: https://d1.awsstatic.com/events/Summits/reinvent2022/CMP334_22986.pdf
[gaudi]: https://habana.ai/wp-content/uploads/pdf/2022/gaudi2-whitepaper.pdf
[grayskull]: https://tenstorrent.com/grayskull/
[m1]: https://www.apple.com/mac/m1/
[m1u]: https://www.apple.com/mac/m1/
[m1max]: https://www.apple.com/mac/m1/
[m2]: https://www.apple.com/mac/m1/
[m2pro]: https://www.apple.com/mac/m1/
