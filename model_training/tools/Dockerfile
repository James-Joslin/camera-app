FROM python:3.12-slim
WORKDIR /workspace/project
ENV HOME=/tmp/model-tools MPLCONFIGDIR=/tmp/matplotlib OPENVINO_TELEMETRY_DISABLED=1
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 libgl1 && rm -rf /var/lib/apt/lists/*
COPY tools/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r /tmp/requirements.txt
ENTRYPOINT ["python", "optimize_model.py"]

