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
    && rm -rf /var/lib/apt/lists/*

# 非 root ユーザーを作成
ARG USERNAME=murasemaru
ARG USER_UID=919
ARG USER_GID=919
RUN groupadd --gid ${USER_GID} ${USERNAME} \
    && useradd --uid ${USER_UID} --gid ${USER_GID} -m ${USERNAME} \
    && apt-get clean

# root 権限で認証情報を処理
RUN --mount=type=secret,id=git_credentials \
    dos2unix -n /run/secrets/.git-credentials /home/${USERNAME}/.git-credentials \
    && chown ${USERNAME}:${USERNAME} /home/${USERNAME}/.git-credentials \
    && chmod 600 /home/${USERNAME}/.git-credentials

# ユーザーを切り替え
USER ${USERNAME}

# 作業ディレクトリを設定
WORKDIR /home/nids-cdd

# パッケージのインストール
COPY .devcontainer/requirements.txt .
RUN --mount=type=cache,target=/home/${USERNAME}/.cache/pip \
    python3 -m pip install -r requirements.txt
