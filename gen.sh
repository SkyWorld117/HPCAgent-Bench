set -euo pipefail
CLAUDE_MODEL=optarena-llm
VLLM_SERVED_MODEL=optarena-vllm
VLLM_API_KEY=EMPTY
VLLM_BASE_URL="http://nid001:8000/v1"
config="$2"
VLLM_REPLICA_URLS="$1"
local_new() {
  local replica
  local -a replicas
  IFS=, read -r -a replicas <<<"${VLLM_REPLICA_URLS:-${VLLM_BASE_URL}}"
  printf 'model_list:\n' >"${config}"
  for replica in "${replicas[@]}"; do
    cat >>"${config}" <<EOF
  - model_name: ${CLAUDE_MODEL:-optarena-llm}
    litellm_params:
      model: hosted_vllm/${VLLM_SERVED_MODEL:-optarena-vllm}
      api_base: ${replica}
      api_key: ${VLLM_API_KEY:-EMPTY}
EOF
  done
  cat >>"${config}" <<EOF
litellm_settings:
  drop_params: true
  set_verbose: false
EOF
  echo "replicas=${#replicas[@]}"
}
local_new
