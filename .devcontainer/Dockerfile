# ベースイメージを設定
ARG PYTHON_VERSION
FROM python:${PYTHON_VERSION}-slim AS base

# ARG の再定義
ARG USERNAME
ARG USER_UID
ARG USER_GID

# 必要なパッケージをインストール
RUN apt-get update && apt-get install -y \
    git \
    dos2unix \
    tcpdump \
    && rm -rf /var/lib/apt/lists/*

# 非 root ユーザーを作成
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && apt-get clean

# root 権限で認証情報を処理
RUN --mount=type=secret,id=git-credentials \
    dos2unix -n /run/secrets/git-credentials /$GUEST_USER_DIR/.git-credentials \
    && chown $USERNAME:$USERNAME /$GUEST_USER_DIR/.git-credentials \
    && chmod 600 /$GUEST_USER_DIR/.git-credentials

# ユーザーを切り替え
USER $USERNAME

# config(git)ファイルの設定
RUN git config --global user.name murasemaru \
    && git config --global user.email bp21071@shibaura-it.ac.jp \
    && git config --global credential.helper store  # 認証ヘルパーを設定

# パッケージのインストール
COPY .devcontainer/requirements.txt /$GUEST_USER_DIR/nids-cdd
RUN --mount=type=cache,target=/$GUEST_USER_DIR/.cache/pip \
    python3 -m pip install -r requirements.txt