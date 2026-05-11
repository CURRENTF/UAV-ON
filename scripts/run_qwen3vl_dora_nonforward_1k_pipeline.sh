#!/usr/bin/env bash
set -euo pipefail

UAV_ROOT="${UAV_ON_ROOT:-/root/autodl-tmp/UAV-ON}"
LVLN_ROOT="${LIGHTNINGVLN_ROOT:-/root/autodl-tmp/LightningVLN}"
LOG_DIR="${LOG_DIR:-/root/autodl-fs/logs}"
MODEL="${QWEN3VL_MODEL_PATH:-/root/autodl-fs/models/Qwen3-VL-4B-Instruct}"
SOURCE_TRAIN_JSONL="${SOURCE_TRAIN_JSONL:-/root/autodl-fs/datasets/uavon_qwen3vl_action_train_astar_20k/train.jsonl}"
REBALANCED_DIR="${REBALANCED_DIR:-/root/autodl-fs/datasets/uavon_qwen3vl_action_train_astar_20k_nonforward80_seed42}"
OUTPUT_DIR="${OUTPUT_DIR:-/root/autodl-fs/checkpoints/uavon_qwen3vl_action_astar_20k_nonforward80_dora_bs4_lr1e-5_linear_1k}"
SMOKE_ADAPTER="${SMOKE_ADAPTER:-/root/autodl-fs/checkpoints/uavon_qwen3vl_action_astar_20k_dora_bs4_e1}"
SOURCE_EVAL_DATASET="${UAV_ON_SOURCE_EVAL_DATASET:-/root/autodl-fs/datasets/uavon_qwen3vl_eval_citypark_slum_20_unique_ids.json}"
EVAL_DATASET="${UAV_ON_EVAL_DATASET:-/root/autodl-fs/datasets/uavon_qwen3vl_eval_citypark_slum_20_unique_ids.json}"
TEST_ENV_ROOT="${UAV_ON_TEST_ENV_ROOT:-/root/autodl-fs/datasets/UAV-ON-test-unpacked}"
EVAL_ROOT="${UAV_ON_EVAL_SAVE_PATH:-/root/autodl-fs/evals/uavon_qwen3vl_action_nonforward80_dora_bs4_lr1e-5_linear_1k_citypark_slum_20}"
SIM_PORT="${UAV_ON_SIM_PORT:-30000}"
MAX_ACTIONS="${UAV_ON_MAX_ACTIONS:-20}"
SERVER_GPUS="${UAV_ON_SERVER_GPUS:-0}"
CUDA_VISIBLE="${UAV_ON_CUDA_VISIBLE_DEVICES:-0}"
RUN_SMOKE="${RUN_SMOKE:-1}"
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_EVAL="${RUN_EVAL:-1}"

ORCH_LOG="$LOG_DIR/qwen3vl_nonforward80_dora_1k_pipeline_orchestrator.log"
TRAIN_LOG="$LOG_DIR/qwen3vl_nonforward80_dora_1k_train.log"
SERVER_LOG="$LOG_DIR/qwen3vl_nonforward80_dora_1k_eval_server.log"
EVAL_LOG="$LOG_DIR/qwen3vl_nonforward80_dora_1k_eval.log"
SUMMARY_LOG="$LOG_DIR/qwen3vl_nonforward80_dora_1k_summary.log"

mkdir -p "$LOG_DIR"

latest_adapter() {
  /root/miniconda3/envs/uavon/bin/python - "$1" <<'PY'
from pathlib import Path
import re
import sys
root = Path(sys.argv[1])
checkpoints = [p for p in root.glob("checkpoint-*") if p.is_dir()]
def step(path):
    match = re.search(r"checkpoint-(\d+)$", path.name)
    return int(match.group(1)) if match else -1
print(max(checkpoints, key=step) if checkpoints else root)
PY
}

require_path() {
  local path="$1"
  if [[ ! -e "$path" ]]; then
    echo "[pipeline] missing required path: $path" | tee -a "$ORCH_LOG"
    exit 1
  fi
}

start_server() {
  /root/miniconda3/envs/uavon/bin/python -u airsim_plugin/AirVLNSimulatorServerTool.py \
    --port "$SIM_PORT" \
    --gpus "$SERVER_GPUS" \
    --root_path "$TEST_ENV_ROOT" \
    > "$SERVER_LOG" 2>&1 &
  SERVER_PID=$!
  echo "[pipeline] server_pid=$SERVER_PID" | tee -a "$ORCH_LOG"
  sleep "${UAV_ON_SERVER_WARMUP_SECONDS:-5}"
}

stop_server() {
  if [[ -n "${SERVER_PID:-}" ]]; then
    kill "$SERVER_PID" 2>/dev/null || true
  fi
  pkill -TERM -f "UAV-ON-test-unpacked/.+Linux-Shipping" 2>/dev/null || true
}

run_eval() {
  local adapter="$1"
  local eval_root="$2"
  local max_actions="$3"
  local adapter_resolved
  adapter_resolved="$(latest_adapter "$adapter")"
  require_path "$adapter_resolved"
  if [[ ! -f "$adapter_resolved/adapter_model.safetensors" || ! -f "$adapter_resolved/adapter_config.json" ]]; then
    echo "[pipeline] adapter is incomplete: $adapter_resolved" | tee -a "$ORCH_LOG"
    exit 1
  fi
  rm -rf "$eval_root"
  mkdir -p "$eval_root"
  cd "$UAV_ROOT"
  start_server
  trap stop_server EXIT
  UAV_ON_VECTOR_ENV_START_METHOD=fork \
  UAV_ON_KINEMATIC_ACTIONS=1 \
  UAV_ON_RGB_ONLY=1 \
  UAV_ON_SCENE_BOOT_SECONDS="${UAV_ON_SCENE_BOOT_SECONDS:-75}" \
  UAV_ON_ACTION_TIMEOUT_SECONDS="${UAV_ON_ACTION_TIMEOUT_SECONDS:-12}" \
  UAV_ON_EVAL_DATASET="$EVAL_DATASET" \
  UAV_ON_EVAL_SAVE_PATH="$eval_root" \
  UAV_ON_CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE" \
  UAV_ON_GPU_ID="${UAV_ON_GPU_ID:-0}" \
  UAV_ON_SIM_PORT="$SIM_PORT" \
  UAV_ON_BATCH_SIZE=1 \
  UAV_ON_MAX_ACTIONS="$max_actions" \
  /root/miniconda3/envs/uavon/bin/python -m src.common.runtime_config \
    --output "$eval_root/runtime_config.launch.json" \
    --scene_boot_seconds_default 75
  UAV_ON_VECTOR_ENV_START_METHOD=fork \
  UAV_ON_KINEMATIC_ACTIONS=1 \
  UAV_ON_RGB_ONLY=1 \
  UAV_ON_SCENE_BOOT_SECONDS="${UAV_ON_SCENE_BOOT_SECONDS:-75}" \
  UAV_ON_ACTION_TIMEOUT_SECONDS="${UAV_ON_ACTION_TIMEOUT_SECONDS:-12}" \
  UAV_ON_EVAL_DATASET="$EVAL_DATASET" \
  UAV_ON_EVAL_SAVE_PATH="$eval_root" \
  UAV_ON_CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE" \
  UAV_ON_GPU_ID="${UAV_ON_GPU_ID:-0}" \
  UAV_ON_SIM_PORT="$SIM_PORT" \
  UAV_ON_BATCH_SIZE=1 \
  UAV_ON_MAX_ACTIONS="$max_actions" \
  QWEN3VL_MODEL_PATH="$MODEL" \
  QWEN3VL_ADAPTER_PATH="$adapter" \
  QWEN3VL_MAX_NEW_TOKENS="${QWEN3VL_MAX_NEW_TOKENS:-8}" \
  QWEN3VL_ATTN_IMPLEMENTATION="${QWEN3VL_ATTN_IMPLEMENTATION:-flash_attention_2}" \
  /root/miniconda3/bin/conda run -n kv --no-capture-output \
    bash scripts/eval_qwen3vl_action.sh \
    > "$EVAL_LOG" 2>&1
  /root/miniconda3/envs/uavon/bin/python scripts/summarize_qwen3vl_action_eval.py \
    --eval_root "$eval_root" \
    --max_actions "$max_actions" \
    --output "$eval_root/metrics_summary.json" \
    > "$SUMMARY_LOG" 2>&1
  stop_server
  trap - EXIT
}

{
  echo "[pipeline] start $(date --iso-8601=seconds)"
  echo "[pipeline] uav_root=$UAV_ROOT"
  echo "[pipeline] lightningvln_root=$LVLN_ROOT"
  echo "[pipeline] model=$MODEL"
  echo "[pipeline] source_train_jsonl=$SOURCE_TRAIN_JSONL"
  echo "[pipeline] rebalanced_dir=$REBALANCED_DIR"
  echo "[pipeline] output_dir=$OUTPUT_DIR"
  echo "[pipeline] eval_dataset=$EVAL_DATASET"
  echo "[pipeline] eval_root=$EVAL_ROOT"
  echo "[pipeline] max_actions=$MAX_ACTIONS"
} | tee "$ORCH_LOG"

for path in "$UAV_ROOT" "$LVLN_ROOT" "$MODEL" "$SOURCE_TRAIN_JSONL" "$SOURCE_EVAL_DATASET" "$EVAL_DATASET" "$TEST_ENV_ROOT"; do
  require_path "$path"
done

if [[ "$RUN_SMOKE" == "1" ]]; then
  echo "[pipeline] smoke eval start $(date --iso-8601=seconds)" | tee -a "$ORCH_LOG"
  UAV_ON_ROOT="$UAV_ROOT" \
  LOG_DIR="$LOG_DIR" \
  QWEN3VL_MODEL_PATH="$MODEL" \
  QWEN3VL_ADAPTER_PATH="$SMOKE_ADAPTER" \
  UAV_ON_SOURCE_EVAL_DATASET="$SOURCE_EVAL_DATASET" \
  UAV_ON_TEST_ENV_ROOT="$TEST_ENV_ROOT" \
  UAV_ON_SIM_PORT="$SIM_PORT" \
  UAV_ON_SERVER_GPUS="$SERVER_GPUS" \
  UAV_ON_CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE" \
  SMOKE_TASKS="${SMOKE_TASKS:-1}" \
  UAV_ON_MAX_ACTIONS="${SMOKE_MAX_ACTIONS:-1}" \
  UAV_ON_EVAL_SAVE_PATH="${SMOKE_EVAL_ROOT:-/root/autodl-fs/evals/qwen3vl_dora_pipeline_smoke}" \
  bash "$UAV_ROOT/scripts/qwen3vl_dora_smoke_eval.sh"
  echo "[pipeline] smoke eval done $(date --iso-8601=seconds)" | tee -a "$ORCH_LOG"
fi

if [[ "$RUN_TRAIN" == "1" ]]; then
  echo "[pipeline] rebalance dataset start $(date --iso-8601=seconds)" | tee -a "$ORCH_LOG"
  cd "$LVLN_ROOT"
  /root/miniconda3/envs/uavon/bin/python scripts/make_qwen3vl_action_rebalanced_dataset.py \
    --input_jsonl "$SOURCE_TRAIN_JSONL" \
    --output_dir "$REBALANCED_DIR" \
    --forward_fraction "${FORWARD_FRACTION:-0.2}" \
    --seed "${REBALANCE_SEED:-42}" \
    --image_mode "${REBALANCE_IMAGE_MODE:-hardlink}" \
    ${REBALANCE_OVERWRITE:+--overwrite} \
    | tee -a "$ORCH_LOG"

  echo "[pipeline] train start $(date --iso-8601=seconds)" | tee -a "$ORCH_LOG"
  MODEL_PATH="$MODEL" \
  DATA_PATH="$REBALANCED_DIR/train.jsonl" \
  OUTPUT_DIR="$OUTPUT_DIR" \
  CACHE_DIR="${CACHE_DIR:-/root/autodl-fs/cache}" \
  PEFT_METHOD=dora \
  MAX_TRAIN_SECONDS=0 \
  MAX_STEPS=1000 \
  PER_DEVICE_TRAIN_BATCH_SIZE=4 \
  GRADIENT_ACCUMULATION_STEPS=1 \
  LEARNING_RATE=1e-5 \
  LR_SCHEDULER_TYPE=linear \
  SAVE_STEPS=200 \
  SAVE_TOTAL_LIMIT=5 \
  LOGGING_STEPS="${LOGGING_STEPS:-10}" \
  REPORT_TO="${REPORT_TO:-wandb}" \
  WANDB_PROJECT="${WANDB_PROJECT:-LightningVLN}" \
  WANDB_NAME="${WANDB_NAME:-uavon_qwen3vl_nonforward80_dora_bs4_lr1e-5_linear_1k}" \
  DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-0}" \
  ATTN_IMPLEMENTATION="${QWEN3VL_ATTN_IMPLEMENTATION:-flash_attention_2}" \
  /root/miniconda3/bin/conda run -n kv --no-capture-output \
    bash "$LVLN_ROOT/scripts/train_qwen3vl_action_baseline.sh" \
    > "$TRAIN_LOG" 2>&1
  echo "[pipeline] train done $(date --iso-8601=seconds)" | tee -a "$ORCH_LOG"
fi

if [[ "$RUN_EVAL" == "1" ]]; then
  echo "[pipeline] final eval start $(date --iso-8601=seconds)" | tee -a "$ORCH_LOG"
  run_eval "$OUTPUT_DIR" "$EVAL_ROOT" "$MAX_ACTIONS"
  /root/miniconda3/envs/uavon/bin/python - <<PY
import json, subprocess
from pathlib import Path
metadata = {
    "model": "$MODEL",
    "adapter_root": "$OUTPUT_DIR",
    "latest_adapter": "$(latest_adapter "$OUTPUT_DIR")",
    "train_jsonl": "$REBALANCED_DIR/train.jsonl",
    "rebalance_stats": "$REBALANCED_DIR/rebalance_stats.json",
    "eval_dataset": "$EVAL_DATASET",
    "eval_root": "$EVAL_ROOT",
    "max_actions": int("$MAX_ACTIONS"),
    "training": {
        "peft_method": "dora",
        "per_device_train_batch_size": 4,
        "gradient_accumulation_steps": 1,
        "learning_rate": 1e-5,
        "lr_scheduler_type": "linear",
        "max_steps": 1000,
        "save_steps": 200,
        "save_total_limit": 5,
    },
    "logs": {
        "orchestrator": "$ORCH_LOG",
        "train": "$TRAIN_LOG",
        "server": "$SERVER_LOG",
        "eval": "$EVAL_LOG",
        "summary": "$SUMMARY_LOG",
    },
}
runtime_config_path = Path("$EVAL_ROOT/runtime_config.json")
launch_runtime_config_path = Path("$EVAL_ROOT/runtime_config.launch.json")
if runtime_config_path.exists():
    metadata["runtime_config"] = json.loads(runtime_config_path.read_text(encoding="utf-8"))
if launch_runtime_config_path.exists():
    metadata["launch_runtime_config"] = json.loads(launch_runtime_config_path.read_text(encoding="utf-8"))
try:
    metadata["uav_on_git_commit"] = subprocess.check_output(["git", "-C", "$UAV_ROOT", "rev-parse", "HEAD"], text=True).strip()
    metadata["lightningvln_git_commit"] = subprocess.check_output(["git", "-C", "$LVLN_ROOT", "rev-parse", "HEAD"], text=True).strip()
except Exception as exc:
    metadata["git_error"] = str(exc)
Path("$EVAL_ROOT/pipeline_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
PY
  echo "[pipeline] final eval done $(date --iso-8601=seconds)" | tee -a "$ORCH_LOG"
fi

echo "[pipeline] done $(date --iso-8601=seconds)" | tee -a "$ORCH_LOG"
