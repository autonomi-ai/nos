ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG PYTHON_VERSION
ENV PYTHON_VERSION=3.8.10
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
ENV CONDA_PATH=/opt/conda/envs/py38
ENV CONDA_EXE=${CONDA_PATH}/bin/conda
ENV PATH=${CONDA_PATH}/bin:$PATH

# Install miniconda
# Install miniconda
RUN curl -o ~/miniconda.sh -O  https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
  chmod +x ~/miniconda.sh && \
  bash ~/miniconda.sh -b -p /opt/conda && \
  rm ~/miniconda.sh && \
  /opt/conda/bin/conda install -y conda-build anaconda-client && \
  /opt/conda/bin/conda create -y -n py38 python=$PYTHON_VERSION

# NOS dependencies
ADD pyproject.toml /tmp/pyproject.toml
ADD requirements /tmp/requirements
RUN cd /tmp/ && \
    pip install --upgrade pip && \
    pip install -r requirements/requirements.txt
ADD nos /tmp/nos
RUN cd /tmp/ && \
    pip install -e . --no-deps

# NOS environment
ENV PROJECT nos
ENV NOS_HOME ~/.nos
ENV NOS_LOGGING_LEVEL INFO
ENV RAY_USAGE_STATS_ENABLED 0
ENV RAY_DATA_DISABLE_PROGRESS_BARS 1
ENV RAY_CONDA_HOME=${CONDA_PATH}

WORKDIR /app/$PROJECT
ENV PYTHONPATH=${PYTHONPATH}:/app/${PROJECT}
RUN echo "export PYTHONPATH=${PYTHONPATH}:/app/${PROJECT}" >> ~/.bashrc

# ENTRYPOINT [ "nos" ]
# CMD [ "serve" ]
