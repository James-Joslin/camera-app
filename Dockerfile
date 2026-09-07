# Final Dockerfile: ML + FastAPI + .NET + utilities
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV DEVELOPMENT_ENV=1

# Install prerequisites
RUN apt-get update && apt-get install -y wget gnupg2 lsb-release software-properties-common curl && rm -rf /var/lib/apt/lists/*

# Add Intel OpenVINO GPG key
RUN wget -qO - https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | apt-key add -

# Add OpenVINO 2025 repository (Ubuntu 24)
RUN echo "deb https://apt.repos.intel.com/openvino ubuntu24 main" | tee /etc/apt/sources.list.d/intel-openvino.list

# Kaggle credentials are mounted at runtime when needed; never bake them into the image.

# --------------------
# Base packages
# --------------------
RUN apt-get update && apt-get install -y \
    software-properties-common \
    apt-utils \
    ca-certificates \
    curl \
    gnupg \
    unzip \
    htop \
    tmux \
    git \
    libicu-dev \
    ffmpeg \
    libsm6 \
    libxext6 \
    build-essential \
    pkg-config \
    wget \
    cmake \
    libopencv-dev \
    pkg-config \
    wget \
    && rm -rf /var/lib/apt/lists/*

RUN update-ca-certificates


RUN apt-get update && apt-get install -y \
    # openvino-runtime-ubuntu22-2023.2 \
    # openvino-dev-ubuntu22-2023.2 \
    intel-opencl-icd \
    openvino-2025.3.0 \
    clinfo \
    intel-gpu-tools \
    && rm -rf /var/lib/apt/lists/*

# --------------------
# Python 3.13 (deadsnakes)
# --------------------
# Install Python 3.13 and required tools
RUN add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get update && apt-get install -y \
        python3.13 \
        python3.13-dev \
        python3.13-venv \
    && rm -rf /var/lib/apt/lists/*

# Ensure pip, setuptools, wheel are installed
RUN python3.13 -m ensurepip && \
    python3.13 -m pip install --upgrade pip setuptools wheel

# Make python3 and pip3 point to Python 3.13
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.13 1 && \
    update-alternatives --install /usr/bin/pip3 pip3 /usr/local/bin/pip 1

# Create a virtual environment for Python packages
RUN python3.13 -m venv /opt/venv

# Activate venv and upgrade pip inside it
RUN /opt/venv/bin/pip install --upgrade pip setuptools wheel

# Ensure python/pip point to the venv by default
ENV PATH="/opt/venv/bin:$PATH"

# --------------------
# Postgres client + ODBC + SQL Server ODBC
# --------------------
RUN apt-get update && apt-get install -y \
    postgresql-client \
    odbc-postgresql \
    && rm -rf /var/lib/apt/lists/*

# MS ODBC for SQL Server
RUN curl https://packages.microsoft.com/keys/microsoft.asc | tee /etc/apt/trusted.gpg.d/microsoft.asc && \
    curl https://packages.microsoft.com/config/ubuntu/22.04/prod.list | tee /etc/apt/sources.list.d/mssql-release.list && \
    apt-get update && ACCEPT_EULA=Y apt-get install -y msodbcsql18 && rm -rf /var/lib/apt/lists/*

# --------------------
# .NET 8 SDK + ASP.NET runtime
# --------------------
RUN apt-get update && apt-get install -y dotnet-sdk-8.0 aspnetcore-runtime-8.0 && rm -rf /var/lib/apt/lists/*

# --------------------
# System utilities (git-lfs)
# --------------------
RUN apt-get update && apt-get install -y git-lfs && rm -rf /var/lib/apt/lists/* || true

# --------------------
# Python libraries (base). Many heavy libs are installed at runtime depending on GPU.
# Installing a core baseline here for faster iteration.
# --------------------
RUN pip3 install --no-cache-dir \
    kaggle \
    fastapi \
    uvicorn[standard] \
    python-multipart \
    pydantic \
    starlette \
    numpy \
    pandas \
    scikit-learn \
    pillow \
    opencv-python \
    matplotlib \
    seaborn \
    plotly \
    tensorboard \
    albumentations \
    torchinfo \
    onnx \
    onnxruntime \
    onnxsim \
    netron \
    mlflow \
    wandb \
    duckdb \
    polars \
    httpx \
    requests \
    python-dotenv \
    watchdog \
    ruff \
    mypy \
    poetry \
    grpcio \
    grpcio-tools \
    azure-storage-blob

# --------------------
# OpenVINO (CPU) install via pip baseline. GPU flavor installed at runtime if needed.
# --------------------
# RUN pip3 install --no-cache-dir openvino openvino-dev  

# --------------------
# Create project dirs and default FastAPI scaffold
# --------------------
WORKDIR /root/project
RUN mkdir -p /root/project/api /root/project/models /root/project/grpc /root/project/data

# Minimal API scaffold (keeps small; you can expand later)
RUN cat > /root/project/api/main.py <<'PY'
from fastapi import FastAPI, UploadFile
from PIL import Image
import io

app = FastAPI()

@app.get('/')
def root():
    return {'status':'ok'}

@app.post('/ping')
async def ping():
    return {'msg':'pong'}
PY

# Example backbone module (user can replace later)
RUN cat > /root/project/models/backbone.py <<'PY'
import torch
import torchvision.models as models

def get_backbone(device='cpu'):
    m = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    backbone = torch.nn.Sequential(
        m.features,
        torch.nn.AdaptiveAvgPool2d((1, 1)),
        torch.nn.Flatten(),
    )
    return backbone.to(device)
PY

# --------------------
# GPU-detection startup script
# - installs proper PyTorch (CPU or CUDA) and OpenVINO GPU packages at container start
# - also installs torchvision/torchaudio matching wheels
# --------------------
RUN cat > /usr/local/bin/startup.sh <<'SH'
#!/usr/bin/env bash
set -e

echo "[startup] Begin runtime setup"

# Ensure pip is available
python3 -m pip --version || (curl -sS https://bootstrap.pypa.io/get-pip.py | python3)

# Lightweight check for NVIDIA drivers
if command -v nvidia-smi &> /dev/null || [ -c /dev/nvidia0 ]; then
    echo "[startup] NVIDIA GPU detected. Installing CUDA-enabled PyTorch and OpenVINO GPU components..."
    # Attempt to install a CUDA wheel (cu121 used as example). If fails, fall back to cpu wheel.
    pip3 install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 \
        pip3 install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    # GPU OpenVINO extras
    pip3 install --no-cache-dir openvino==2025.3.0
else
    echo "[startup] No GPU detected. Installing (or ensuring) CPU-only PyTorch..."
    pip3 install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu 
    pip3 install --no-cache-dir openvino==2025.3.0
fi

# Make sure common tools are available
pip3 install --no-cache-dir jupyterlab

echo "[startup] Runtime setup complete. Executing command: $@"
exec "$@"
SH

RUN chmod +x /usr/local/bin/startup.sh

# --------------------
# Environment variables (as requested)
# --------------------
ENV POSTGRES_HOST=db
ENV POSTGRES_PORT=5432
ENV POSTGRES_DB=cameras
ENV AZURITE_BLOB_ENDPOINT=http://azurite:10000/devstoreaccount1
ENV AZURITE_ACCOUNT_NAME=devstoreaccount1
ENV AZURITE_DATA_CONTAINER=computer-vision-data
ENV AZURITE_MODEL_CONTAINER=computer-vision-models
ENV USE_AZURITE=true


ENV DEVELOPMENT_ENV=1

# .NET WebAPI default ports
ENV ASPNETCORE_URLS="http://0.0.0.0:5000;https://0.0.0.0:5001"

# -------------------
# Install Node.js & npm
# -------------------
# Use NodeSource for latest Node.js LTS
RUN curl -fsSL https://deb.nodesource.com/setup_lts.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# --------------------
# Expose common ports (FastAPI, JupyterLab, TensorBoard, .NET WebAPI)
# --------------------
EXPOSE 8000 8888 6006 5000 5001

# ENTRYPOINT runs the startup script which will finish by executing the CMD
ENTRYPOINT ["/usr/local/bin/startup.sh"]

# Default command: keep shell so user can run services interactively.
# For automatic server start, override CMD to: ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
# Set up the environment for interactive bash use
RUN echo "PS1='\w\$ '" > /root/.bashrc

# Keep container running
CMD tail -f /dev/null
