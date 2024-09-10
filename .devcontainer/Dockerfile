# ベースイメージを設定
ARG PYTHON_VERSION=3.11.9
FROM python:${PYTHON_VERSION}-slim AS base

# Dockerfile内のコマンドを実行するコンテナ内のディレクトリ=WORKDIRの設定
ARG WORKDIR=/home
WORKDIR ${WORKDIR}

# ホストのディレクトリをコンテナ内にCOPY
COPY .. .

# 必要なパッケージをインストール
RUN apt-get update && apt-get install -y \
    git \
    curl \
    jq \
    && rm -rf /var/lib/apt/lists/*

# Pythonの依存関係をインストール
COPY .devcontainer/requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install -r requirements.txt

# スクリプトをコンテナ内にコピー
COPY .devcontainer/entrypoint.sh /usr/local/bin/entrypoint.sh