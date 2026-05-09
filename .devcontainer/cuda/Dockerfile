FROM docker.io/nvidia/cuda:12.6.3-runtime-ubuntu24.04 AS cuda-base

FROM mcr.microsoft.com/devcontainers/python:3.12

ARG CLAUDE_CODE_VERSION=latest
COPY --from=ghcr.io/astral-sh/uv:0.11.2 /uv /uvx /usr/local/bin/

# Copy CUDA from official image
COPY --from=cuda-base /usr/local/cuda /usr/local/cuda

RUN apt-get update && apt-get install -y --no-install-recommends \
    libopus0 \
    nodejs \
    npm \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && chmod 1777 /tmp

ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Ensure vscode user has access to npm global dir
RUN mkdir -p /usr/local/share/npm-global && \
    chown -R vscode:vscode /usr/local/share/npm-global

USER vscode

# Install Claude Code globally via npm
ENV NPM_CONFIG_PREFIX=/usr/local/share/npm-global
ENV PATH=$PATH:/usr/local/share/npm-global/bin
RUN npm install -g @anthropic-ai/claude-code@${CLAUDE_CODE_VERSION}
