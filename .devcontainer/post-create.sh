#!/usr/bin/env bash
set -euxo pipefail

# Install dependencies
uv sync --dev

# Write Claude Code settings so it picks up Bedrock env vars
mkdir -p ~/.claude
cat > ~/.claude/settings.json <<EOF
{
  "env": {
    "ENABLE_TOOL_SEARCH": "${ENABLE_TOOL_SEARCH}",
    "CLAUDE_CODE_EFFORT_LEVEL": "${CLAUDE_CODE_EFFORT_LEVEL}",
    "CLAUDE_CODE_DISABLE_ADAPTIVE_THINKING": "${CLAUDE_CODE_DISABLE_ADAPTIVE_THINKING}",
    "MAX_THINKING_TOKENS": "${MAX_THINKING_TOKENS}",
    "CLAUDE_CODE_USE_BEDROCK": "${CLAUDE_CODE_USE_BEDROCK}",
    "ANTHROPIC_MODEL": "${ANTHROPIC_MODEL}",
    "ANTHROPIC_SMALL_FAST_MODEL": "${ANTHROPIC_SMALL_FAST_MODEL}",
    "AWS_PROFILE": "${AWS_PROFILE}"
  }
}
EOF

# Install peon-ping (auto-detects devcontainer and routes audio to host relay)
curl -fsSL https://raw.githubusercontent.com/PeonPing/peon-ping/main/install.sh | bash

# Install Claude Code Usage Monitor
git clone https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor.git ~/.claude-usage-monitor \
  && cd ~/.claude-usage-monitor \
  && uv tool install .
