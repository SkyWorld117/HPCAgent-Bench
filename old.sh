set -euo pipefail
CLAUDE_MODEL=optarena-llm
VLLM_SERVED_MODEL=optarena-vllm
VLLM_API_KEY=EMPTY
VLLM_BASE_URL="http://nid001:8000/v1"
config="$1"
cat >"${config}" <<EOF
model_list:
  - model_name: ${CLAUDE_MODEL:-optarena-llm}
    litellm_params:
      model: hosted_vllm/${VLLM_SERVED_MODEL:-optarena-vllm}
      api_base: ${VLLM_BASE_URL}
      api_key: ${VLLM_API_KEY:-EMPTY}
litellm_settings:
  drop_params: true
  set_verbose: false
EOF
