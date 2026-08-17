#!/usr/bin/env bash
set -euo pipefail

# Configure the primary DeepGraph LLM route without putting the credential in
# argv, shell history, the protected legacy worktree, or a public web route.
# This script deliberately does not restart services: the operator must first
# run the normal no-active-work deployment gate.

if [[ ${EUID} -ne 0 ]]; then
  echo "Run as root: sudo $0" >&2
  exit 2
fi

runtime_dir=/etc/deepgraph
runtime_env=${runtime_dir}/runtime.env
web_dropin=/etc/systemd/system/deepgraph-web.service.d/20-llm-runtime.conf
worker_dropin=/etc/systemd/system/deepgraph-auto-execute.service.d/20-llm-runtime.conf

read -r -p "Base URL [https://api.deepseek.com]: " base_url
base_url=${base_url:-https://api.deepseek.com}
read -r -p "Model [deepseek-v4-flash]: " model
model=${model:-deepseek-v4-flash}
read -r -p "Protocol [chat_completions]: " protocol
protocol=${protocol:-chat_completions}
case ${protocol} in
  chat_completions|responses) ;;
  *) echo "Protocol must be chat_completions or responses." >&2; exit 2 ;;
esac

read -r -s -p "API key (input hidden): " api_key
echo
if [[ -z ${api_key} || ${api_key} == *$'\n'* || ${api_key} == *$'\r'* ]]; then
  echo "A single-line, non-empty API key is required." >&2
  exit 2
fi

read -r -p "Verify GET /models before saving? [Y/n]: " verify
verify=${verify:-Y}
if [[ ${verify} =~ ^[Yy]$ ]]; then
  models_url=${base_url%/}/models
  echo "Checking ${models_url} ..."
  # Feed the credential through curl's stdin config so it never appears in
  # the process argument list.
  if ! curl --config - <<EOF
url = "${models_url}"
header = "Authorization: Bearer ${api_key}"
fail
silent
show-error
max-time = 20
output = "/dev/null"
EOF
  then
    echo "Provider canary failed; no configuration was changed." >&2
    exit 1
  fi
fi

quote_env() {
  local value=${1}
  value=${value//\\/\\\\}
  value=${value//\"/\\\"}
  printf '"%s"' "${value}"
}

install -d -m 0750 -o root -g root "${runtime_dir}"
if [[ -f ${runtime_env} ]]; then
  backup=${runtime_env}.bak.$(date -u +%Y%m%dT%H%M%SZ)
  install -m 0600 -o root -g root "${runtime_env}" "${backup}"
  echo "Previous runtime environment backed up to ${backup}"
fi

tmp_env=$(mktemp "${runtime_dir}/runtime.env.XXXXXX")
cleanup() { rm -f "${tmp_env}"; }
trap cleanup EXIT
{
  printf 'DEEPGRAPH_LLM_API_KEY=%s\n' "$(quote_env "${api_key}")"
  printf 'DEEPGRAPH_LLM_BASE_URL=%s\n' "$(quote_env "${base_url}")"
  printf 'DEEPGRAPH_LLM_MODEL=%s\n' "$(quote_env "${model}")"
  printf 'DEEPGRAPH_LLM_PROTOCOL=%s\n' "$(quote_env "${protocol}")"
} >"${tmp_env}"
chmod 0600 "${tmp_env}"
chown root:root "${tmp_env}"
mv -f "${tmp_env}" "${runtime_env}"
trap - EXIT

install -D -m 0644 /dev/stdin "${web_dropin}" <<'EOF'
[Service]
EnvironmentFile=-/etc/deepgraph/runtime.env
EOF
install -D -m 0644 /dev/stdin "${worker_dropin}" <<'EOF'
[Service]
EnvironmentFile=-/etc/deepgraph/runtime.env
EOF

systemctl daemon-reload
echo "Configured ${model} in ${runtime_env}; the key was not printed."
echo "No service was restarted. After the no-active-work gate, restart and verify:"
echo "  sudo systemctl restart deepgraph-web.service deepgraph-auto-execute.service"
echo "  curl --fail http://127.0.0.1:8080/api/health/data"
