#!/bin/bash
# Start the ChoriProx proxy with all API keys loaded.
# Usage: ./start.sh [port]
# Default port: 6666

set -euo pipefail
cd "$(dirname "$0")"

PORT="${1:-6666}"

# Remote backend API keys
export OPENCODE_API_KEY="${OPENCODE_API_KEY:-}"
export MIMO_API_KEY="${MIMO_API_KEY:-}"
export MINIMAX_API_KEY="${MINIMAX_API_KEY:-}"
export OLLAMA_API_KEY="${OLLAMA_API_KEY:-}"
export KILO_API_KEY="${KILO_API_KEY:-}"
export ARK_MODEL_ARK_API_KEY="${ARK_MODEL_ARK_API_KEY:-}"
export DEEPSEEK_API_KEY="${DEEPSEEK_API_KEY:-}"
export AGNES_API_KEY="${AGNES_API_KEY:-}"
if [ -z "${PROXY_API_KEY:-}" ]; then
    if [ -f "data/proxy_api_key.txt" ]; then
        export PROXY_API_KEY="$(cat data/proxy_api_key.txt)"
    else
        export PROXY_API_KEY="$(openssl rand -hex 32)"
        mkdir -p data
        echo "$PROXY_API_KEY" > data/proxy_api_key.txt
        chmod 600 data/proxy_api_key.txt
    fi
fi
export NEO4J_AUTH="${NEO4J_AUTH:-neo4j:your-neo4j-password}"

# StepFun API key (SOTA backend)
export STEP_FUN_API_KEY="${STEP_FUN_API_KEY:-your-stepfun-api-key-here}"
export BAILIAN_TOKEN_PLAN_API_KEY="${BAILIAN_TOKEN_PLAN_API_KEY:-}"
export DASHSCOPE_INTL_API_KEY="${DASHSCOPE_INTL_API_KEY:-}"

# SOTA extraction backend (StepFun)
export PROXY_SOTA_API_KEY="${PROXY_SOTA_API_KEY:-$STEP_FUN_API_KEY}"
export PROXY_SOTA_ENDPOINT="${PROXY_SOTA_ENDPOINT:-https://api.stepfun.ai/step_plan/v1}"
export PROXY_SOTA_MODEL="${PROXY_SOTA_MODEL:-step-3.7-flash}"
export PROXY_SOTA_MODE="${PROXY_SOTA_MODE:-unified}"
export PROXY_SOTA_TIMEOUT="${PROXY_SOTA_TIMEOUT:-180}"
export PROXY_SOTA_SUMMARY_GATE="${PROXY_SOTA_SUMMARY_GATE:-true}"

echo "Starting ChoriProx on port $PORT..."
for credential_name in OPENCODE_API_KEY MIMO_API_KEY STEP_FUN_API_KEY \
    ARK_MODEL_ARK_API_KEY BAILIAN_TOKEN_PLAN_API_KEY DASHSCOPE_INTL_API_KEY \
    PROXY_API_KEY NEO4J_AUTH; do
    if [ -n "${!credential_name}" ]; then
        printf '  %s: configured\n' "$credential_name"
    else
        printf '  %s: not configured\n' "$credential_name"
    fi
done

PYTHONPATH=src exec python3 src/qwen_proxy.py
