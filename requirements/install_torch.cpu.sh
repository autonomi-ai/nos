#!/bin/bash
set -ex

echo "Building CPU version of torch"
mamba install -yv pytorch==2.0.1 torchvision cpuonly -c pytorch;
echo "Building CPU version of torch done"
python -c "import torch as t; print(f'torch={t.__version__}, cuda={t.cuda.is_available()}, cudnn={t.backends.cudnn.is_available()}')"
