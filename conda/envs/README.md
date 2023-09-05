# NOS conda environments

1. Create and activate a new environment.

```bash
conda create -n <env_name> python=3.8
conda activate <env_name>
```

Optionally, you can build the new environment from an existing environment file:
```bash
conda env create -n <env_name> -f conda/envs/base-gpu-cu118/base-gpu-cu118.yml
```

2. Install packages

Install the necessary packages from conda and PyPI.

```bash
pip install ... # PyPI packages
```

3. Export the environment

```bash
conda env export -n <env_name> --no-builds > conda/envs/<env_name>/<env_name>.yml
```


## Supported conda environments

Here's a table of all the conda environments we support:

| Environment | Python | CUDA | Conda | Description |
| ----------- | ------ | ---- | ----- | ----------- |
| `base-cpu` | 3.8 | - | [base-cpu.yml](conda/envs/base-cpu/base-cpu.yml) | Base environment for CPU inference |
| `base-gpu-cu118` | 3.8 | 11.8 | [base-gpu-cu118.yml](conda/envs/base-gpu-cu118/base-gpu-cu118.yml) | Base environment for GPU inference |
| `base-neuron` | 3.8 | - | [base-neuron.yml](conda/envs/base-neuron/base-neuron.yml) | Base environment for Neuron inference |
| `whisperx` | 3.8 | 11.8 | [whisperx.yml](conda/envs/whisperx/whisperx-cu118.yml) | Environment for [WhisperX](https://github.com/m-bain/whisperX) inference |
| `diffusers-latest` | 3.8 | 11.8 | [diffusers-latest](nos/server/train/dreambooth/config.py) | Environment for [Dreambooth LoRA fine-tuning](https://github.com/autonomi-ai/nos/blob/main/nos/models/dreambooth/dreambooth.py#L138) |
| `mmdetection-latest` | 3.8 | 11.8 | [mmdetection-latest](nos/server/train/config.py) | Environment for [OpenMMLab MMDetection fine-tuning](nos/server/train/config.py) |
