ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG TARGET
ARG PYTHON_VERSION
ENV PYTHON_VERSION=3.8.10
ENV PYENV=py38_${TARGET}
ENV DEBIAN_FRONTEND="noninteractive"
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV PYTHONWARNINGS ignore
ENV LOGURU_LEVEL WARNING
ENV TZ="America/Los_Angeles"

# Pre-requisites for installing miniconda
RUN apt-get -y update && \
  apt-get install -y gnupg2 sudo curl tzdata && \
  apt-get -y autoclean && \
  apt-get -y autoremove && \
  rm -rf /var/lib/apt/lists/*

# Setup conda paths (see RAY_CONDA_HOME dependency below)
ENV CONDA_PATH=/opt/conda/envs/${PYENV}
ENV CONDA_EXE=${CONDA_PATH}/bin/conda
ENV PATH=${CONDA_PATH}/bin:$PATH

# Install miniconda
COPY ./conda/envs/base-${TARGET}/env.yml /tmp/conda/envs/base-${TARGET}/env.yml
RUN curl -o ~/miniconda.sh -O  https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
  chmod +x ~/miniconda.sh && \
  bash ~/miniconda.sh -b -p /opt/conda && \
  rm ~/miniconda.sh && \
  /opt/conda/bin/conda init bash && \
  /opt/conda/bin/conda env create -n ${PYENV} -f /tmp/conda/envs/base-${TARGET}/env.yml

# NOS dependencies
ENV PROJECT nos
WORKDIR /app/$PROJECT
ADD pyproject.toml .
ADD requirements requirements
RUN pip install --upgrade pip && \
    pip install -r requirements/requirements.txt
ADD nos nos

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

ENTRYPOINT ["nos"]
CMD ["serve"]
