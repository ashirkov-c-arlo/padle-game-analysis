FROM mcr.microsoft.com/devcontainers/python:3.12

ARG CLAUDE_CODE_VERSION=latest

COPY --from=ghcr.io/astral-sh/uv:0.11.2 /uv /uvx /usr/local/bin/

RUN apt-get update && apt-get install -y --no-install-recommends \
    libopus0 \
    nodejs \
    npm \
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
