ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG TARGET
ENV PYENV=py3_${TARGET}
ENV DEBIAN_FRONTEND="noninteractive"
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV PYTHONWARNINGS ignore
ENV LOGURU_LEVEL WARNING
ENV TZ="America/Los_Angeles"

# Pre-requisites for installing miniconda
RUN apt-get -y update \
  && apt-get install -y --no-install-recommends gnupg2 sudo curl tzdata \
  && apt-get -y autoclean \
  && apt-get -y autoremove \
  && rm -rf /var/lib/apt/lists/*

# Setup conda paths (see RAY_CONDA_HOME dependency below)
ENV CONDA_PATH=/opt/conda/envs/${PYENV}
ENV CONDA_EXE=${CONDA_PATH}/bin/conda
ENV PATH=${CONDA_PATH}/bin:$PATH

# Install mambaforge
COPY ./conda/envs/base-${TARGET}/env.yml /tmp/conda/envs/base-${TARGET}/env.yml
RUN curl -sLo ~/mambaforge.sh https://github.com/conda-forge/miniforge/releases/download/4.12.0-2/Mambaforge-4.12.0-2-Linux-x86_64.sh \
  && chmod +x ~/mambaforge.sh \
  && ~/mambaforge.sh -b -p /opt/conda \
  && /opt/conda/bin/mamba init bash \
  && /opt/conda/bin/mamba env create -n ${PYENV} -f /tmp/conda/envs/base-${TARGET}/env.yml \
  && rm ~/mambaforge.sh \
  && rm /tmp/conda/envs/base-${TARGET}/env.yml  \
  && /opt/conda/bin/mamba clean -ya \
  && rm -rf /opt/conda/pkgs/*

# NOS dependencies
ENV PROJECT nos
WORKDIR /app/$PROJECT
ADD pyproject.toml .
ADD requirements requirements
RUN pip install -r requirements/requirements.txt \
  && rm -rf ~/.cache/pip
ADD nos nos
RUN pip install -e . --no-deps

# NOS environment
ENV NOS_HOME /app/.nos
ENV NOS_LOGGING_LEVEL INFO

ENV RAY_USAGE_STATS_ENABLED 0
ENV RAY_DATA_DISABLE_PROGRESS_BARS 1
ENV RAY_CONDA_HOME=${CONDA_PATH}
ENV TORCH_HOME ${NOS_HOME}/cache/torch
ENV TRANSFORMERS_CACHE ${NOS_HOME}/cache/transformers
ENV HF_HOME ${NOS_HOME}/cache/transformers

ENV PYTHONPATH=${PYTHONPATH}:/app/${PROJECT}
RUN echo "export PYTHONPATH=${PYTHONPATH}:/app/${PROJECT}" >> ~/.bashrc
RUN echo "export PATH=/opt/conda/envs/${PYENV}/bin:$PATH" >> ~/.bashrc
