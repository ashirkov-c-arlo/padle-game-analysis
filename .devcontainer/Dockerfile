FROM mcr.microsoft.com/devcontainers/python:3.12

ARG CLAUDE_CODE_VERSION=latest

COPY --from=ghcr.io/astral-sh/uv:0.11.2 /uv /uvx /usr/local/bin/

# ROCm runtime repo
RUN curl -fsSL https://repo.radeon.com/rocm/rocm.gpg.key | gpg --dearmor -o /etc/apt/keyrings/rocm.gpg && \
    echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/${ROCM_VERSION} noble main" \
      > /etc/apt/sources.list.d/rocm.list && \
    printf 'Package: *\nPin: release o=repo.radeon.com\nPin-Priority: 600\n' \
      > /etc/apt/preferences.d/rocm-pin-600

RUN apt-get update && apt-get install -y --no-install-recommends \
    libopus0 \
    nodejs \
    npm \
    # ROCm runtime + training stack
    rocminfo \
    rocm-smi-lib \
    hip-runtime-amd \
    rocblas \
    hipblas \
    rocsolver \
    hipsolver \
    miopen-hip \
    rccl \
    rocfft \
    rocrand \
    hipfft \
    hipsparse \
    roctracer \
    rocprofiler-register \
    && rm -rf /var/lib/apt/lists/* \
    && chmod 1777 /tmp


# Ensure vscode user has access to npm global dir
RUN mkdir -p /usr/local/share/npm-global && \
    chown -R vscode:vscode /usr/local/share/npm-global
USER vscode

# Install Claude Code globally via npm
ENV NPM_CONFIG_PREFIX=/usr/local/share/npm-global
ENV PATH=$PATH:/usr/local/share/npm-global/bin
RUN npm install -g @anthropic-ai/claude-code@${CLAUDE_CODE_VERSION}

USER vscode
