FROM docker.io/rocm/dev-ubuntu-24.04:6.2.4 AS rocm-base

FROM mcr.microsoft.com/devcontainers/python:3.12

ARG CLAUDE_CODE_VERSION=latest
COPY --from=ghcr.io/astral-sh/uv:0.11.2 /uv /uvx /usr/local/bin/

# Copy ROCm from official image
COPY --from=rocm-base /opt/rocm /opt/rocm

RUN apt-get update && apt-get install -y --no-install-recommends \
    libopus0 \
    nodejs \
    npm \
    libelf1 \
    libnuma1 \
    libdrm2 \
    libdrm-amdgpu1 \
    && rm -rf /var/lib/apt/lists/* \
    && chmod 1777 /tmp

ENV PATH="/opt/rocm/bin:${PATH}"
ENV LD_LIBRARY_PATH="/opt/rocm/lib:${LD_LIBRARY_PATH}"
ENV HSA_OVERRIDE_GFX_VERSION=10.3.0
ENV PYTORCH_ROCM_ARCH=gfx1030

# Ensure vscode user has access to npm global dir
RUN mkdir -p /usr/local/share/npm-global && \
    chown -R vscode:vscode /usr/local/share/npm-global

RUN groupadd -f video && groupadd -f render
USER vscode

# Install Claude Code globally via npm
ENV NPM_CONFIG_PREFIX=/usr/local/share/npm-global
ENV PATH=$PATH:/usr/local/share/npm-global/bin
RUN npm install -g @anthropic-ai/claude-code@${CLAUDE_CODE_VERSION}
