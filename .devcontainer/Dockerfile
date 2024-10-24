# ベースイメージを設定
ARG PYTHON_VERSION=3.11.9
FROM python:${PYTHON_VERSION}-slim AS base

# Dockerfile内のコマンドを実行するコンテナ内のディレクトリ=WORKDIRの設定
#ARG WORKDIR=/home
#WORKDIR ${WORKDIR}
#
## ホストのディレクトリをコンテナ内にCOPY
#COPY .. .

# 必要なパッケージをインストール
RUN apt-get update && apt-get install -y \
    git \
    dos2unix \
    libpcap-dev \
    && rm -rf /var/lib/apt/lists/* ?

# config(git)ファイルの設定
RUN git config --global user.name murasemaru \
    && git config --global user.email bp21071@shibaura-it.ac.jp

COPY .devcontainer/requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install -r requirements.txt