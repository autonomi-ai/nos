# Dockerfile for NOS
# Targets:
#  - base  : base image for nos dependencies (client/server)
#  - server: base + nos
#  - test  : base + nos + tests
#
# >>>>>>>>>>>>
ARG BASE_IMAGE
FROM ${BASE_IMAGE} as base

ARG TARGET
ENV PROJECT nos
ENV PYENV=${PROJECT}_${TARGET}
ENV DEBIAN_FRONTEND="noninteractive"
ENV PYTHON_VERSION 3.8.10
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV PYTHONWARNINGS ignore
ENV LOGURU_LEVEL WARNING
ENV TZ="America/Los_Angeles"

# Pre-requisites for installing miniconda
RUN apt-get -y update \
  && apt-get install -y --no-install-recommends gnupg2 sudo curl tzdata build-essential \
  && apt-get -y autoclean \
  && apt-get -y autoremove \
  && rm -rf /var/lib/apt/lists/*

# Setup conda paths (see RAY_CONDA_HOME dependency below)
ENV CONDA_PATH=/opt/conda/envs/${PYENV}
ENV CONDA_PREFIX=${CONDA_PATH}
ENV CONDA_EXE=${CONDA_PATH}/bin/conda
ENV PATH=${CONDA_PATH}/bin:/opt/conda/bin:$PATH
ENV CONDA_DEFAULT_ENV ${PYENV}

# Install mambaforge
RUN curl -sLo ~/mambaforge.sh "https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-$(uname)-$(uname -m).sh" \
  && chmod +x ~/mambaforge.sh \
  && ~/mambaforge.sh -b -p /opt/conda \
  && /opt/conda/bin/mamba init bash \
  && /opt/conda/bin/mamba create -n ${PYENV} python=${PYTHON_VERSION} -y \
  && rm ~/mambaforge.sh

# NOS dependencies and install
WORKDIR /tmp/$PROJECT
ADD requirements requirements

# Install torch (cpu/gpu) based on target
RUN --mount=type=cache,target=/opt/conda/pkgs/ \
  if [ "${TARGET}" = "cpu" ]; then \
    mamba install -yv pytorch torchvision cpuonly -c pytorch; \
  else \
    mamba install -yv pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia; \
  fi

# Install client/server dependencies
RUN --mount=type=cache,target=~/.cache/pip \
  pip install \
  -r requirements/requirements.txt \
  -r requirements/requirements.server.txt

# Cleanup
RUN /opt/conda/bin/mamba clean -ya \
  && rm -rf /opt/conda/pkgs/* \
  && rm -rf ~/.cache/pip \
  && rm -rf /tmp/$PROJECT

# NOS environment
ENV NOS_HOME=/app/.nos
ENV NOS_LOGGING_LEVEL=WARNING
ENV NOS_MAX_CONCURRENT_MODELS=1

# Torch, HF, Transformers cache
ENV TORCH_HOME=${NOS_HOME}/cache/torch
ENV HF_HOME=${NOS_HOME}/cache/transformers
ENV TRANSFORMERS_CACHE=${NOS_HOME}/cache/transformers

# Ray environment
ENV RAY_LOG_TO_STDERR=0
ENV RAY_DEDUP_LOGS=0
ENV RAY_USAGE_STATS_ENABLED=0
ENV RAY_DATA_DISABLE_PROGRESS_BARS=1
ENV RAY_CONDA_HOME=${CONDA_PATH}

RUN echo "export CONDA_PATH=/opt/conda/envs/${PYENV}" >> ~/.bashrc
RUN echo "export PATH=/opt/conda/envs/${PYENV}/bin:$PATH" >> ~/.bashrc
RUN echo "export CONDA_DEFAULT_ENV=${PYENV}" >> ~/.bashrc
RUN echo "mamba activate ${PYENV}" > ~/.bashrc

# >>>>>>>>>>>>>>>>>>>>
# Install NOS / server
# >>>>>>>>>>>>>>>>>>>>
FROM base as server

WORKDIR /tmp/$PROJECT
ADD pyproject.toml .
ADD nos nos
RUN pip install . --no-deps && \
  rm -rf ~/.cache/pip && \
  rm -rf /tmp/$PROJECT

WORKDIR /app/$PROJECT
ENV NOS_ENV=${PYENV}

CMD ["nos-grpc-server"]

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>
# Install NOS test environment
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>
FROM server as test

RUN apt-get update \
  && apt-get install make
ADD pyproject.toml .
ADD nos nos
ADD requirements requirements
ADD makefiles makefiles
ADD Makefile .
ADD tests tests
RUN pip install -e '.[test]'
