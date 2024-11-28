# ベースイメージを設定
ARG PYTHON_VERSION=3.11.9
FROM python:${PYTHON_VERSION}-slim AS base

# Python環境変数の設定
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# 必要なパッケージをインストール
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    dos2unix \
    tcpdump \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Gitの設定
RUN git config --global user.name "murasemaru" \
    && git config --global user.email "bp21071@shibaura-it.ac.jp"

# 非rootユーザーを作成
RUN useradd -m -s /bin/bash murasemaru

# 作業ディレクトリを作成し、ユーザーを切り替える
WORKDIR /app
USER murasemaru

# 必要なPythonライブラリをインストール
COPY --chown=appuser:appuser .devcontainer/requirements.txt .
RUN --mount=type=cache,target=/home/appuser/.cache/pip \
    python3 -m pip install --no-cache-dir -r requirements.txt

# デフォルトの作業ディレクトリ
COPY --chown=appuser:appuser . /app

# デフォルトのエントリポイント（必要に応じて変更可能）
CMD ["python3"]
