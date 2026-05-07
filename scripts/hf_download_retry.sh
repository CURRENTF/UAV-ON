#!/usr/bin/env bash
set -u

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <repo-id> <repo-type> <local-dir>" >&2
    exit 2
fi

repo_id="$1"
repo_type="$2"
local_dir="$3"
retry_delay="${HF_DOWNLOAD_RETRY_DELAY:-60}"

mkdir -p "$local_dir"

while true; do
    echo "[$(date '+%F %T')] starting: hf download $repo_id --repo-type $repo_type --local-dir $local_dir"

    if bash -ic 'unproxy' >/dev/null 2>&1; then
        echo "[$(date '+%F %T')] ran unproxy"
    else
        echo "[$(date '+%F %T')] unproxy command was unavailable; unsetting proxy variables directly"
    fi
    unset http_proxy https_proxy ftp_proxy all_proxy no_proxy
    unset HTTP_PROXY HTTPS_PROXY FTP_PROXY ALL_PROXY NO_PROXY
    export HF_ENDPOINT=https://hf-mirror.com

    if hf download "$repo_id" --repo-type "$repo_type" --local-dir "$local_dir"; then
        echo "[$(date '+%F %T')] completed: $repo_id"
        exit 0
    fi

    status=$?
    echo "[$(date '+%F %T')] failed with exit code $status: $repo_id"
    echo "[$(date '+%F %T')] retrying in ${retry_delay}s"
    sleep "$retry_delay"
done
