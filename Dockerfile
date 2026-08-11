# Build:  docker build -t obscura .
# CLI:    docker run --rm -v "$PWD:/data" obscura run /data/in.mp4 -o /data/out.mp4
# Web UI: docker run --rm -p 8000:8000 obscura serve --host 0.0.0.0
FROM python:3.12-slim AS base

# opencv-python-headless still needs libGL's companions for video I/O.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UNIFACE_HOME=/models

WORKDIR /app

# Dependency layer first, so source edits do not re-resolve the wheel set.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir '.[cpu,web]'

# Weights download on first use; give them a stable home that can be mounted.
RUN mkdir -p /models /data
VOLUME ["/models"]
WORKDIR /data

ENTRYPOINT ["obscura"]
CMD ["--help"]
