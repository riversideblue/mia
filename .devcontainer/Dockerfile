# ベースイメージを設定
ARG PYTHON_VERSION=3.11.9
FROM python:${PYTHON_VERSION}-slim AS base

# Python環境変数の設定
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
    
# 必要なパッケージをインストール
RUN apt-get update && apt-get install -y \
    git \
    dos2unix \
    tcpdump \
    && rm -rf /var/lib/apt/lists/* ?

# config(git)ファイルの設定
RUN git config --global user.name murasemaru \
    && git config --global user.email bp21071@shibaura-it.ac.jp

# 非rootユーザーを作成
ARG USERNAME=murasemaru
ARG USER_UID=919
ARG USER_GID=919

RUN groupadd --gid ${USER_GID} ${USERNAME} \
    && useradd --uid ${USER_UID} --gid ${USER_GID} -m ${USERNAME} \
    && apt-get clean

WORKDIR /home/nids-cdd
# ユーザーを切り替え
USER ${USERNAME}

COPY .devcontainer/requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install -r requirements.txt
