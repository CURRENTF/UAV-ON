#!/usr/bin/env bash
set -euo pipefail

ROOT="${UAV_ON_ROOT:-/root/autodl-tmp/UAV-ON}"
MODEL="${QWEN3VL_MODEL_PATH:-/root/autodl-fs/models/Qwen3-VL-4B-Instruct}"
ADAPTER="${QWEN3VL_ADAPTER_PATH-}"
DATASET="${UAV_ON_EVAL_DATASET:-/root/autodl-fs/datasets/uavon_qwen3vl_eval_full_valset_1000_unique_ids.json}"
TEST_ENV_ROOT="${UAV_ON_TEST_ENV_ROOT:-/root/autodl-fs/datasets/UAV-ON-test-unpacked}"
RUN_ID="${RUN_ID:-qwen3vl_front_rgb_full_valset_$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-/root/autodl-fs/logs/$RUN_ID}"
EVAL_ROOT="${UAV_ON_EVAL_SAVE_PATH:-/root/autodl-fs/evals/$RUN_ID}"
SIM_PORT="${UAV_ON_SIM_PORT:-30001}"
SERVER_GPUS="${UAV_ON_SERVER_GPUS:-0}"
CUDA_VISIBLE="${UAV_ON_CUDA_VISIBLE_DEVICES:-0}"
GPU_ID="${UAV_ON_GPU_ID:-0}"
MAX_ACTIONS="${UAV_ON_MAX_ACTIONS:-40}"
SCENE_BOOT_SECONDS="${UAV_ON_SCENE_BOOT_SECONDS:-300}"
ACTION_TIMEOUT_SECONDS="${UAV_ON_ACTION_TIMEOUT_SECONDS:-12}"
SAMPLE_MODE="${QWEN3VL_EVAL_SAMPLE_MODE:-single}"
TRAJECTORY_MAX_STEPS="${QWEN3VL_TRAJECTORY_MAX_STEPS:-0}"
TRAJECTORY_KV_CACHE="${QWEN3VL_TRAJECTORY_KV_CACHE:-false}"
MAX_NEW_TOKENS="${QWEN3VL_MAX_NEW_TOKENS:-8}"

ORCH_LOG="$LOG_DIR/orchestrator.log"
SERVER_LOG="$LOG_DIR/server.log"
EVAL_LOG="$LOG_DIR/eval.log"
SUMMARY_LOG="$LOG_DIR/summary.log"

mkdir -p "$LOG_DIR" "$EVAL_ROOT"
cd "$ROOT"

for path in "$MODEL" "$DATASET" "$TEST_ENV_ROOT"; do
  if [[ ! -e "$path" ]]; then
    echo "[eval] missing required path: $path" | tee -a "$ORCH_LOG"
    exit 1
  fi
done

latest_adapter=""
if [[ -n "$ADAPTER" ]]; then
  latest_adapter="$(
    /root/miniconda3/envs/uavon/bin/python - <<PY
from pathlib import Path
import re
root = Path("$ADAPTER")
checkpoints = [p for p in root.glob("checkpoint-*") if p.is_dir()]
def step(path):
    match = re.search(r"checkpoint-(\\d+)$", path.name)
    return int(match.group(1)) if match else -1
print(max(checkpoints, key=step) if checkpoints else root)
PY
  )"
  if [[ ! -f "$latest_adapter/adapter_model.safetensors" || ! -f "$latest_adapter/adapter_config.json" ]]; then
    echo "[eval] adapter is incomplete: $latest_adapter" | tee -a "$ORCH_LOG"
    exit 1
  fi
fi

{
  echo "[eval] start $(date --iso-8601=seconds)"
  echo "[eval] root=$ROOT"
  echo "[eval] git=$(git rev-parse HEAD)"
  echo "[eval] model=$MODEL"
  echo "[eval] adapter_root=${ADAPTER:-<none>}"
  echo "[eval] latest_adapter=${latest_adapter:-<none>}"
  echo "[eval] dataset=$DATASET"
  echo "[eval] eval_root=$EVAL_ROOT"
  echo "[eval] max_actions=$MAX_ACTIONS"
  echo "[eval] sample_mode=$SAMPLE_MODE"
  echo "[eval] trajectory_max_steps=$TRAJECTORY_MAX_STEPS"
  echo "[eval] trajectory_kv_cache=$TRAJECTORY_KV_CACHE"
  echo "[eval] max_new_tokens=$MAX_NEW_TOKENS"
  echo "[eval] sim_port=$SIM_PORT"
  echo "[eval] scene_ready_timeout_seconds=$SCENE_BOOT_SECONDS"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
} | tee "$ORCH_LOG"

/root/miniconda3/envs/uavon/bin/python -u airsim_plugin/AirVLNSimulatorServerTool.py \
  --port "$SIM_PORT" \
  --gpus "$SERVER_GPUS" \
  --root_path "$TEST_ENV_ROOT" \
  > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!

cleanup() {
  kill "$SERVER_PID" 2>/dev/null || true
  pkill -TERM -f "UAV-ON-test-unpacked/.+Linux-Shipping" 2>/dev/null || true
}
trap cleanup EXIT

echo "[eval] server_pid=$SERVER_PID" | tee -a "$ORCH_LOG"
sleep "${UAV_ON_SERVER_WARMUP_SECONDS:-5}"

UAV_ON_VECTOR_ENV_START_METHOD=fork \
UAV_ON_KINEMATIC_ACTIONS=1 \
UAV_ON_RGB_ONLY=1 \
UAV_ON_FRONT_VIEW_ONLY=1 \
UAV_ON_SCENE_BOOT_SECONDS="$SCENE_BOOT_SECONDS" \
UAV_ON_ACTION_TIMEOUT_SECONDS="$ACTION_TIMEOUT_SECONDS" \
UAV_ON_EVAL_DATASET="$DATASET" \
UAV_ON_EVAL_SAVE_PATH="$EVAL_ROOT" \
UAV_ON_CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE" \
UAV_ON_GPU_ID="$GPU_ID" \
UAV_ON_SIM_PORT="$SIM_PORT" \
UAV_ON_BATCH_SIZE=1 \
UAV_ON_MAX_ACTIONS="$MAX_ACTIONS" \
QWEN3VL_MODEL_PATH="$MODEL" \
QWEN3VL_ADAPTER_PATH="$ADAPTER" \
QWEN3VL_MAX_NEW_TOKENS="$MAX_NEW_TOKENS" \
QWEN3VL_EVAL_SAMPLE_MODE="$SAMPLE_MODE" \
QWEN3VL_TRAJECTORY_MAX_STEPS="$TRAJECTORY_MAX_STEPS" \
QWEN3VL_TRAJECTORY_KV_CACHE="$TRAJECTORY_KV_CACHE" \
QWEN3VL_ATTN_IMPLEMENTATION="${QWEN3VL_ATTN_IMPLEMENTATION:-flash_attention_2}" \
/root/miniconda3/bin/conda run -n kv --no-capture-output \
  bash scripts/eval_qwen3vl_action.sh \
  > "$EVAL_LOG" 2>&1

/root/miniconda3/envs/uavon/bin/python scripts/summarize_qwen3vl_action_eval.py \
  --eval_root "$EVAL_ROOT" \
  --max_actions "$MAX_ACTIONS" \
  --output "$EVAL_ROOT/metrics_summary.json" \
  > "$SUMMARY_LOG" 2>&1

/root/miniconda3/envs/uavon/bin/python - <<PY
import json
import subprocess
from pathlib import Path

metadata = {
    "model": "$MODEL",
    "adapter_root": "$ADAPTER" or None,
    "latest_adapter": "$latest_adapter" or None,
    "dataset": "$DATASET",
    "eval_root": "$EVAL_ROOT",
    "max_actions": int("$MAX_ACTIONS"),
    "batch_size": 1,
    "view_mode": "front_rgb",
    "sample_mode": "$SAMPLE_MODE",
    "trajectory_max_steps": int("$TRAJECTORY_MAX_STEPS"),
    "trajectory_kv_cache": "$TRAJECTORY_KV_CACHE".lower() in {"1", "true", "yes", "on"},
    "max_new_tokens": int("$MAX_NEW_TOKENS"),
    "action_mode": "kinematic",
    "rgb_only": True,
    "front_view_only": True,
    "sim_port": int("$SIM_PORT"),
    "scene_boot_seconds": float("$SCENE_BOOT_SECONDS"),
    "scene_ready_timeout_seconds": float("$SCENE_BOOT_SECONDS"),
    "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "logs": {
        "orchestrator": "$ORCH_LOG",
        "server": "$SERVER_LOG",
        "eval": "$EVAL_LOG",
        "summary": "$SUMMARY_LOG",
    },
}
Path("$EVAL_ROOT/benchmark_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
PY

echo "[eval] done $(date --iso-8601=seconds)" | tee -a "$ORCH_LOG"
echo "[eval] summary=$EVAL_ROOT/metrics_summary.json" | tee -a "$ORCH_LOG"
