# NexRL Multi-Stage Dockerfile
# Base on NVIDIA NGC PyTorch image

# Stage 1: Builder - Build wheel package
ARG NGC_IMAGE_TAG=25.10-py3

FROM nvcr.io/nvidia/pytorch:${NGC_IMAGE_TAG} AS builder

LABEL stage="builder"
LABEL description="NexRL Builder Stage - Build wheel package"

WORKDIR /build

# 安装构建依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# 升级 pip 和安装构建工具
RUN pip install --no-cache-dir --upgrade pip setuptools wheel build

# 复制项目文件
COPY pyproject.toml /build/
COPY README.md /build/
COPY nexrl/ /build/nexrl/
COPY cli/ /build/cli/

# 构建 wheel 包
RUN python -m build --wheel --outdir /build/dist

# 验证 wheel 包
RUN ls -lh /build/dist/*.whl

# Stage 2: Runtime - Production Image
ARG NGC_IMAGE_TAG=25.10-py3

FROM nvcr.io/nvidia/pytorch:${NGC_IMAGE_TAG} AS runtime

# 元数据
LABEL maintainer="NexAGI Team"
LABEL description="NexRL Universal Image - Supports Controller, Workers, and Train Router"
LABEL version="1.0.0"
LABEL base.image="nvcr.io/nvidia/pytorch:${NGC_IMAGE_TAG}"

# 构建参数传递到运行时环境变量
ARG NGC_IMAGE_TAG
ENV NGC_IMAGE_TAG=${NGC_IMAGE_TAG}

# 设置工作目录
WORKDIR /app

# 安装额外的系统依赖（NGC 镜像已包含大部分工具）
RUN apt-get update && apt-get install -y --no-install-recommends \
    vim \
    && rm -rf /var/lib/apt/lists/*

# 升级 pip 和基础工具
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# 从 builder 阶段复制 wheel 包
COPY --from=builder /build/dist/*.whl /tmp/

# 安装 NexRL wheel 包（包含所有可选依赖）
# 使用 [all] 安装所有 optional-dependencies（training, flash, tracking, distributed, math, apis）
RUN WHEEL_FILE=$(ls /tmp/*.whl) && \
    pip install --no-cache-dir "${WHEEL_FILE}[all]" && \
    rm -rf /tmp/*.whl

# 从 builder 阶段复制配置文件到容器
COPY --from=builder /build/nexrl/config /app/config

# 设置 Python 环境变量
ENV PYTHONPATH=/app:$PYTHONPATH
ENV PYTHONUNBUFFERED=1

# 创建必要的目录
RUN mkdir -p /app/data /app/checkpoints /app/logs
