# 使用 Python 基础镜像
FROM python:3.12.12-slim

# 配置 uv 使用系统 Python 和国内镜像
ENV UV_PYTHON_PREFERENCE=only-system
ENV UV_SYSTEM_PYTHON=1
ENV UV_INDEX_URL=https://mirrors.cloud.tencent.com/pypi/simple



FROM nvidia/cuda:12.6.3-runtime-ubuntu22.04

WORKDIR /home

# 设置非交互模式
ENV DEBIAN_FRONTEND=noninteractive \
    PIPX_HOME=/opt/pipx \
    PIPX_BIN_DIR=/usr/local/bin 

# 设置清华源
RUN sed -i 's/archive.ubuntu.com/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list && \
    sed -i 's/security.ubuntu.com/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list


RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    pipx \
    git \
    vim \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libc6 \
    libopenblas-dev \
    libgl1 \
    libglu1-mesa \
    libegl1 \ 
    wget \
    && rm -rf /var/lib/apt/lists/*

# RUN apt-get update && apt-get install -y python3.12 python3.12-venv

RUN pipx ensurepath

RUN pipx install uv

# 1. 安装必要工具
RUN apt-get update && apt-get install -y \
    git \
    openssh-client \
    bash \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /root/.ssh && \
    chmod 700 /root/.ssh

SHELL ["/bin/bash", "-c"]

RUN mkdir /home/logs
ENV UV_INDEX_URL=https://mirrors.cloud.tencent.com/pypi/simple
RUN --mount=type=secret,id=known_hosts,target=/root/.ssh/known_hosts \
    --mount=type=ssh \
    git clone --recurse-submodules git@[YOUR-GIT-SERVER]:ai/chatscreenshotanalysisserver.git 
RUN --mount=type=secret,id=known_hosts,target=/root/.ssh/known_hosts \
    --mount=type=ssh \
    cd /home/chatscreenshotanalysisserver && \
    git pull


COPY ./models /home/model_hub/paddle/models
ENV PADDLE_MODEL_DIR=/home/model_hub/paddle/
RUN --mount=type=secret,id=known_hosts,target=/root/.ssh/known_hosts \
    --mount=type=ssh \
    cd /home/chatscreenshotanalysisserver && chmod +x start.sh && ./start.sh


ENV DEBIAN_FRONTEND=dialog
