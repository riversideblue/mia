ARG PYTHON_VERSION=3.11.9
FROM python:${PYTHON_VERSION}-slim AS base

# 必要なパッケージをインストール
RUN apt-get update && apt-get install -y \
    git \
    dos2unix \
    tcpdump \
    && rm -rf /var/lib/apt/lists/*

# ARG の再定義
ARG USERNAME=murasemaru
ARG USER_UID=919
ARG USER_GID=919

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 非 root ユーザーを作成
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && apt-get clean

USER $USERNAME
# requirements.txt をコピーしてパッケージをインストール
COPY .devcontainer/requirements.txt /tmp/requirements.txt
RUN python3 -m pip install --no-cache-dir -r /tmp/requirements.txt
