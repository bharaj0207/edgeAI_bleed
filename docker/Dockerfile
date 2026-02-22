FROM python:3.11-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    QNN_SDK_ROOT=/opt/qairt

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    build-essential \
    ca-certificates \
    cmake \
    curl \
    git \
    libglib2.0-0 \
    libgomp1 \
    libstdc++6 \
    libusb-1.0-0 \
    pkg-config \
    unzip \
    wget \
    xz-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY pyproject.toml README.md /workspace/
COPY src /workspace/src
COPY configs /workspace/configs
COPY docs /workspace/docs
COPY scripts /workspace/scripts
COPY tests /workspace/tests
COPY docker /workspace/docker

RUN python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install -e '.[dev]'

RUN chmod +x /workspace/docker/*.sh /workspace/scripts/*.sh 2>/dev/null || true

ENTRYPOINT ["/workspace/docker/entrypoint.sh"]
CMD ["bash"]
