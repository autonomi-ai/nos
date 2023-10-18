#!/bin/bash
set -ex

echo "Building GPU version of torch (CUDA_VERSION=${CUDA_VERSION})"
mamba install -yv pytorch==2.0.1 torchvision torchaudio pytorch-cuda=${CUDA_VERSION} -c pytorch -c nvidia;
echo "Building GPU version of torch done"
python -c "import torch as t; print(f'torch={t.__version__}, cuda={t.cuda.is_available()}, cudnn={t.backends.cudnn.is_available()}')"
