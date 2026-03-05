# =============================================================================
# Face Recognition Project - Docker Build
# Supports both CPU and GPU (NVIDIA CUDA) deployments
#
# Build:
#   CPU:  docker build -t face-recognition:cpu .
#   GPU:  docker build -t face-recognition:gpu --build-arg BASE_IMAGE=nvidia/cuda:12.6.0-runtime-ubuntu22.04 --build-arg ONNX_PKG=onnxruntime-gpu .
#
# Run:
#   CPU:  docker run -p 8000:8000 -v ./models:/app/models -v ./data:/app/data face-recognition:cpu
#   GPU:  docker run --gpus all -p 8000:8000 -v ./models:/app/models -v ./data:/app/data face-recognition:gpu
# =============================================================================

ARG BASE_IMAGE=ubuntu:22.04

# -----------------------------------------------------------------------------
# Stage 1: Builder — compile C++ module and install Python packages
# -----------------------------------------------------------------------------
FROM ubuntu:22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev python3-venv \
    g++ cmake make \
    libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .

# Install Python dependencies into a venv
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY cpp/ cpp/
COPY python/ python/
COPY build.sh .

# Remove Windows-specific compiled files
RUN rm -f python/*.pyd python/*.lib python/*.exp

# Build the C++ pybind11 module for Linux
RUN bash build.sh

# -----------------------------------------------------------------------------
# Stage 2: Runtime
# -----------------------------------------------------------------------------
FROM ${BASE_IMAGE} AS runtime

ARG ONNX_PKG=onnxruntime

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-venv \
    libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the venv from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install the correct onnxruntime variant (cpu or gpu)
RUN pip install --no-cache-dir ${ONNX_PKG}

# Copy application code and compiled C++ module
COPY --from=builder /app/python/ python/
COPY --from=builder /app/build.sh .
COPY --from=builder /app/cpp/ cpp/
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

# Create mount point directories
RUN mkdir -p models data/known_faces

EXPOSE 8000
ENTRYPOINT ["./entrypoint.sh"]
