FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System packages needed by scientific/python stack
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    sqlite3 \
    libsqlite3-0 \
    libsqlite3-dev \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Fail early if the Python sqlite extension cannot load. This protects notebook
# and Jupyter-adjacent workflows that import sqlite during kernel startup.
RUN python -c "import sqlite3; print('sqlite:', sqlite3.sqlite_version)"

# Install CUDA-capable PyTorch wheels.
# On GPU hosts, `torch.cuda.is_available()` becomes true when running with
# NVIDIA runtime (`--gpus all` / compose gpu profile).
RUN pip install --upgrade pip wheel "setuptools==80.9.0" \
    && pip install --index-url https://download.pytorch.org/whl/cu121 \
       torch==2.4.1 torchvision==0.19.1

# Install ROS-free project dependencies
COPY requirements_docker.txt /tmp/requirements_docker.txt
RUN pip install -r /tmp/requirements_docker.txt

# Install Tianshou from a pinned source revision for reproducibility
ARG TIANSHOU_REPO=https://github.com/thu-ml/tianshou.git
ARG TIANSHOU_REF=935a85a09fed1466379e26378c11821c6a7c9954
RUN git clone "${TIANSHOU_REPO}" /tmp/tianshou \
    && git -C /tmp/tianshou checkout "${TIANSHOU_REF}" \
    && pip install /tmp/tianshou \
    && rm -rf /tmp/tianshou

COPY . /app

# Live Plotly viewer and JupyterLab ports
EXPOSE 8765 8888

# Useful default shell; override with docker run ... <command>
CMD ["bash"]
