ARG PYTHON_VERSION=3.11.9
FROM ubuntu:20.04 AS base

# Prevents Python from writing pyc files.
ENV PYTHONDONTWRITEBYTECODE=1

# Keeps Python from buffering stdout and stderr to avoid situations where
# the application crashes without emitting any logs due to buffering.
ENV PYTHONUNBUFFERED=1

# 必要なパッケージをインストール
RUN apt-get update && apt-get install -y \
    git \
    curl \
    jq \
    python3=${PYTHON_VERSION}* \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Create a non-privileged user that the app will run under.
ARG UID=10001
RUN adduser \
    --gecos "" \
    --uid "${UID}" \
    black

# Read and install Python dependencies
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install -r requirements.txt

# Switch to the non-privileged user to run the application.
USER black

# This setting is for development. It should be better "/app"

WORKDIR /home/black

# Copy the source code into the container.
COPY . .

# Expose the port that the application listens on.
EXPOSE 8000

# スクリプトをコンテナ内にコピー
COPY start.sh /usr/local/bin/start.sh

# Run the application.
CMD ["/usr/local/bin/start.sh"]
