#!/usr/bin/env bash
set -euo pipefail

ROOT="${UAV_ON_ROOT:-/root/autodl-tmp/UAV-ON}"
LOG_DIR="${LOG_DIR:-/root/autodl-fs/logs}"
MODEL="${QWEN3VL_MODEL_PATH:-/root/autodl-fs/models/Qwen3-VL-4B-Instruct}"
ADAPTER="${QWEN3VL_ADAPTER_PATH:-/root/autodl-fs/checkpoints/uavon_qwen3vl_action_astar_20k_dora_bs4_e1}"
SOURCE_DATASET="${UAV_ON_SOURCE_EVAL_DATASET:-/root/autodl-fs/datasets/uavon_qwen3vl_eval_citypark_slum_20_unique_ids.json}"
SMOKE_TASKS="${SMOKE_TASKS:-1}"
DATASET="${UAV_ON_EVAL_DATASET:-/root/autodl-fs/datasets/uavon_qwen3vl_eval_smoke_${SMOKE_TASKS}.json}"
TEST_ENV_ROOT="${UAV_ON_TEST_ENV_ROOT:-/root/autodl-fs/datasets/UAV-ON-test-unpacked}"
EVAL_ROOT="${UAV_ON_EVAL_SAVE_PATH:-/root/autodl-fs/evals/qwen3vl_dora_smoke_eval}"
SIM_PORT="${UAV_ON_SIM_PORT:-30000}"
MAX_ACTIONS="${UAV_ON_MAX_ACTIONS:-1}"
SERVER_GPUS="${UAV_ON_SERVER_GPUS:-0}"
CUDA_VISIBLE="${UAV_ON_CUDA_VISIBLE_DEVICES:-0}"

ORCH_LOG="$LOG_DIR/qwen3vl_dora_smoke_eval_orchestrator.log"
SERVER_LOG="$LOG_DIR/qwen3vl_dora_smoke_eval_server.log"
EVAL_LOG="$LOG_DIR/qwen3vl_dora_smoke_eval.log"
SUMMARY_LOG="$LOG_DIR/qwen3vl_dora_smoke_eval_summary.log"

mkdir -p "$LOG_DIR" "$EVAL_ROOT"
cd "$ROOT"

/root/miniconda3/envs/uavon/bin/python - <<PY
import json
from pathlib import Path
source = Path("$SOURCE_DATASET")
target = Path("$DATASET")
tasks = int("$SMOKE_TASKS")
data = json.loads(source.read_text(encoding="utf-8"))
if not isinstance(data, list) or not data:
    raise ValueError(f"Invalid or empty source eval dataset: {source}")
target.parent.mkdir(parents=True, exist_ok=True)
target.write_text(json.dumps(data[:tasks], indent=2, ensure_ascii=False), encoding="utf-8")
print(f"[smoke] wrote {target} rows={min(len(data), tasks)}")
PY

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

for path in "$MODEL" "$latest_adapter" "$DATASET" "$TEST_ENV_ROOT"; do
  if [[ ! -e "$path" ]]; then
    echo "[smoke] missing required path: $path" | tee -a "$ORCH_LOG"
    exit 1
  fi
done

if [[ ! -f "$latest_adapter/adapter_model.safetensors" || ! -f "$latest_adapter/adapter_config.json" ]]; then
  echo "[smoke] adapter is incomplete: $latest_adapter" | tee -a "$ORCH_LOG"
  exit 1
fi

{
  echo "[smoke] start $(date --iso-8601=seconds)"
  echo "[smoke] model=$MODEL"
  echo "[smoke] adapter_root=$ADAPTER"
  echo "[smoke] latest_adapter=$latest_adapter"
  echo "[smoke] dataset=$DATASET"
  echo "[smoke] eval_root=$EVAL_ROOT"
  echo "[smoke] max_actions=$MAX_ACTIONS"
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

echo "[smoke] server_pid=$SERVER_PID" | tee -a "$ORCH_LOG"
sleep "${UAV_ON_SERVER_WARMUP_SECONDS:-5}"

UAV_ON_VECTOR_ENV_START_METHOD=fork \
UAV_ON_KINEMATIC_ACTIONS=1 \
UAV_ON_RGB_ONLY=1 \
UAV_ON_SCENE_BOOT_SECONDS="${UAV_ON_SCENE_BOOT_SECONDS:-75}" \
UAV_ON_ACTION_TIMEOUT_SECONDS="${UAV_ON_ACTION_TIMEOUT_SECONDS:-12}" \
UAV_ON_EVAL_DATASET="$DATASET" \
UAV_ON_EVAL_SAVE_PATH="$EVAL_ROOT" \
UAV_ON_CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE" \
UAV_ON_GPU_ID="${UAV_ON_GPU_ID:-0}" \
UAV_ON_SIM_PORT="$SIM_PORT" \
UAV_ON_BATCH_SIZE=1 \
UAV_ON_MAX_ACTIONS="$MAX_ACTIONS" \
/root/miniconda3/envs/uavon/bin/python -m src.common.runtime_config \
  --output "$EVAL_ROOT/runtime_config.launch.json" \
  --scene_boot_seconds_default 75

UAV_ON_VECTOR_ENV_START_METHOD=fork \
UAV_ON_KINEMATIC_ACTIONS=1 \
UAV_ON_RGB_ONLY=1 \
UAV_ON_SCENE_BOOT_SECONDS="${UAV_ON_SCENE_BOOT_SECONDS:-75}" \
UAV_ON_ACTION_TIMEOUT_SECONDS="${UAV_ON_ACTION_TIMEOUT_SECONDS:-12}" \
UAV_ON_EVAL_DATASET="$DATASET" \
UAV_ON_EVAL_SAVE_PATH="$EVAL_ROOT" \
UAV_ON_CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE" \
UAV_ON_GPU_ID="${UAV_ON_GPU_ID:-0}" \
UAV_ON_SIM_PORT="$SIM_PORT" \
UAV_ON_BATCH_SIZE=1 \
UAV_ON_MAX_ACTIONS="$MAX_ACTIONS" \
QWEN3VL_MODEL_PATH="$MODEL" \
QWEN3VL_ADAPTER_PATH="$ADAPTER" \
QWEN3VL_MAX_NEW_TOKENS="${QWEN3VL_MAX_NEW_TOKENS:-8}" \
QWEN3VL_ATTN_IMPLEMENTATION="${QWEN3VL_ATTN_IMPLEMENTATION:-flash_attention_2}" \
/root/miniconda3/bin/conda run -n kv --no-capture-output \
  bash scripts/eval_qwen3vl_action.sh \
  > "$EVAL_LOG" 2>&1

/root/miniconda3/envs/uavon/bin/python scripts/summarize_qwen3vl_action_eval.py \
  --eval_root "$EVAL_ROOT" \
  --max_actions "$MAX_ACTIONS" \
  --output "$EVAL_ROOT/metrics_summary.json" \
  > "$SUMMARY_LOG" 2>&1

echo "[smoke] done $(date --iso-8601=seconds)" | tee -a "$ORCH_LOG"
echo "[smoke] summary=$EVAL_ROOT/metrics_summary.json" | tee -a "$ORCH_LOG"
