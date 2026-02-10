#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  download-huggingface-dataset.sh --dataset <org/name> [options]

Options:
  --dataset <org/name>   Hugging Face dataset repo (required)
  --output <dir>         Output directory (default: dataset)
  --revision <rev>       Dataset revision/branch/tag (default: main)
  --num-workers <n>      Parallel download workers (default: 8)
  --retries <n>          Number of retry attempts on failure (default: 3)
  --retry-delay <sec>    Delay between retries in seconds (default: 10)
  --enable-xet           Enable Xet transfer backend
  --disable-xet          Disable Xet transfer backend (default)
  --append-date          Append YYYY-MM-DD to output directory
  --foreground           Run in foreground (default is detached nohup)
  -h, --help             Show this help
EOF
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Error: required command '$1' not found in PATH." >&2
    exit 1
  fi
}

download_dataset() {
  local dataset_name="$1"
  local output_dir="$2"
  local revision="$3"
  local num_workers="$4"
  local retries="$5"
  local retry_delay="$6"
  local disable_xet="$7"

  if [[ -e "${output_dir}" && ! -d "${output_dir}" ]]; then
    echo "Error: output path exists and is not a directory: ${output_dir}" >&2
    exit 1
  fi

  if [[ -d "${output_dir}" ]]; then
    echo "Output directory exists; resuming download into: ${output_dir}"
  fi
  mkdir -p "${output_dir}"

  if [[ "${disable_xet}" == "true" ]]; then
    export HF_HUB_DISABLE_XET=1
  fi

  local attempt=1
  while (( attempt <= retries )); do
    echo "Started download attempt ${attempt}/${retries} at $(date '+%Y-%m-%d %H:%M:%S')"
    if python3 - "${dataset_name}" "${output_dir}" "${revision}" "${num_workers}" <<'PY'
import os
import socket
import sys

try:
    from huggingface_hub import snapshot_download
except ImportError as exc:
    print(
        "Error: huggingface_hub is not installed. Install with: pip install huggingface_hub",
        file=sys.stderr,
    )
    raise SystemExit(1) from exc

dataset_name = sys.argv[1]
output_dir = sys.argv[2]
revision = sys.argv[3]
num_workers = int(sys.argv[4])
if num_workers < 1:
    print("Error: --num-workers must be >= 1", file=sys.stderr)
    raise SystemExit(1)

token = os.environ.get("HF_TOKEN")

try:
    socket.getaddrinfo("huggingface.co", 443)
except socket.gaierror as exc:
    print(f"Error: DNS/network cannot resolve huggingface.co: {exc}", file=sys.stderr)
    raise SystemExit(2) from exc

snapshot_download(
    repo_id=dataset_name,
    repo_type="dataset",
    revision=revision,
    local_dir=output_dir,
    max_workers=num_workers,
    token=token,
)

print("Finished snapshot download")
PY
    then
      echo "Finished download at $(date '+%Y-%m-%d %H:%M:%S')"
      return 0
    fi

    if (( attempt == retries )); then
      echo "Download failed after ${retries} attempts." >&2
      return 1
    fi
    echo "Attempt ${attempt} failed; retrying in ${retry_delay}s..." >&2
    sleep "${retry_delay}"
    attempt=$((attempt + 1))
  done
}

append_date=false
foreground=false
dataset_name=""
output="dataset"
revision="main"
num_workers="8"
retries="3"
retry_delay="10"
disable_xet="true"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset)
      [[ $# -ge 2 ]] || { echo "Error: --dataset requires a value." >&2; usage; exit 1; }
      dataset_name="$2"
      shift 2
      ;;
    --output)
      [[ $# -ge 2 ]] || { echo "Error: --output requires a value." >&2; usage; exit 1; }
      output="$2"
      shift 2
      ;;
    --revision)
      [[ $# -ge 2 ]] || { echo "Error: --revision requires a value." >&2; usage; exit 1; }
      revision="$2"
      shift 2
      ;;
    --num-workers)
      [[ $# -ge 2 ]] || { echo "Error: --num-workers requires a value." >&2; usage; exit 1; }
      num_workers="$2"
      shift 2
      ;;
    --retries)
      [[ $# -ge 2 ]] || { echo "Error: --retries requires a value." >&2; usage; exit 1; }
      retries="$2"
      shift 2
      ;;
    --retry-delay)
      [[ $# -ge 2 ]] || { echo "Error: --retry-delay requires a value." >&2; usage; exit 1; }
      retry_delay="$2"
      shift 2
      ;;
    --enable-xet)
      disable_xet="false"
      shift
      ;;
    --disable-xet)
      disable_xet="true"
      shift
      ;;
    --append-date)
      append_date=true
      shift
      ;;
    --foreground)
      foreground=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Error: unknown argument '$1'" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "${dataset_name}" ]]; then
  echo "Error: --dataset is required." >&2
  usage
  exit 1
fi

if ! [[ "${num_workers}" =~ ^[0-9]+$ ]] || [[ "${num_workers}" -lt 1 ]]; then
  echo "Error: --num-workers must be a positive integer." >&2
  exit 1
fi

if ! [[ "${retries}" =~ ^[0-9]+$ ]] || [[ "${retries}" -lt 1 ]]; then
  echo "Error: --retries must be a positive integer." >&2
  exit 1
fi

if ! [[ "${retry_delay}" =~ ^[0-9]+$ ]] || [[ "${retry_delay}" -lt 0 ]]; then
  echo "Error: --retry-delay must be a non-negative integer." >&2
  exit 1
fi

require_cmd python3

if [[ "${append_date}" == "true" ]]; then
  output="${output}-$(date '+%Y-%m-%d')"
fi

logfile="${output}.log"
echo "Downloading: ${dataset_name}"
echo "Output dir:  ${output}"
echo "Revision:    ${revision}"
echo "Workers:     ${num_workers}"
echo "Retries:     ${retries}"
echo "Retry delay: ${retry_delay}s"
if [[ "${disable_xet}" == "true" ]]; then
  echo "Xet:         disabled"
else
  echo "Xet:         enabled"
fi
echo "Log file:    ${logfile}"
if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN:    not set (public datasets only)"
else
  echo "HF_TOKEN:    set"
fi

if [[ "${foreground}" == "true" ]]; then
  download_dataset "${dataset_name}" "${output}" "${revision}" "${num_workers}" "${retries}" "${retry_delay}" "${disable_xet}" 2>&1 | tee "${logfile}"
else
  nohup "$0" --foreground --dataset "${dataset_name}" --output "${output}" --revision "${revision}" --num-workers "${num_workers}" --retries "${retries}" --retry-delay "${retry_delay}" $([[ "${disable_xet}" == "true" ]] && echo "--disable-xet" || echo "--enable-xet") >"${logfile}" 2>&1 &
  echo "Started in background with PID: $!"
  echo "Monitor progress with: tail -f ${logfile}"
fi
