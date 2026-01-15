# NexRL Multi-Stage Dockerfile
# Based on NVIDIA NGC PyTorch image

# Stage 1: Builder - Build wheel package
ARG NGC_IMAGE_TAG=25.02-py3

FROM nvcr.io/nvidia/pytorch:${NGC_IMAGE_TAG} AS builder

ARG MAX_JOBS=16

LABEL stage="builder"
LABEL description="NexRL Builder Stage - Build wheel package"

WORKDIR /build

# Set MAX_JOBS for parallel compilation
ENV MAX_JOBS=${MAX_JOBS}

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install build tools
RUN pip install --no-cache-dir --upgrade pip setuptools wheel build

# Copy project files
COPY pyproject.toml /build/
COPY README.md /build/
COPY nexrl/ /build/nexrl/
COPY cli/ /build/cli/

# Build wheel package
RUN python -m build --wheel --outdir /build/dist

# Verify wheel package
RUN ls -lh /build/dist/*.whl

# Stage 2: Runtime - Production Image
ARG NGC_IMAGE_TAG=25.02-py3

FROM nvcr.io/nvidia/pytorch:${NGC_IMAGE_TAG} AS runtime

# Metadata
LABEL maintainer="NexAGI Team"
LABEL description="NexRL Universal Image - Supports Controller, Workers, and Train Router"
LABEL version="1.0.0"

# Pass build arguments to runtime environment variables
ARG NGC_IMAGE_TAG
ARG MAX_JOBS=16
ENV NGC_IMAGE_TAG=${NGC_IMAGE_TAG}
ENV MAX_JOBS=${MAX_JOBS}

# Set working directory
WORKDIR /app

# Install additional system dependencies (NGC image already includes most tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    vim \
    tmux \
    tini \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and basic tools
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy wheel package from builder stage
COPY --from=builder /build/dist/*.whl /tmp/

# Install NexRL wheel package with all optional dependencies
# Uses [all] to install all optional-dependencies (dev, advanced-tracking)
# This includes nex-weaver, tinker, and tinker-cookbook from GitHub
RUN WHEEL_FILE=$(ls /tmp/*.whl) && \
    pip install --no-cache-dir "${WHEEL_FILE}[all]" && \
    rm -rf /tmp/*.whl

RUN pip uninstall -y flash-attn && \
pip install --no-cache-dir flash-attn --no-build-isolation

# Set Python environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Create necessary directories
RUN mkdir -p /app/data /app/checkpoints /app/logs

# Use tini as the init system to handle signals properly
ENTRYPOINT ["/usr/bin/tini", "--"]
