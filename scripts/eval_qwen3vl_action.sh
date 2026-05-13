#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
root_dir=.
data_root="${UAV_ON_DATA_ROOT:-/data2/haojitai/datasets/uav-on}"
dataset_path="${UAV_ON_EVAL_DATASET:-$data_root/DATASET/merged_valset.json}"
cuda_visible_devices="${UAV_ON_CUDA_VISIBLE_DEVICES:-0}"
gpu_id="${UAV_ON_GPU_ID:-0}"
sim_port="${UAV_ON_SIM_PORT:-30000}"
eval_save_path="${UAV_ON_EVAL_SAVE_PATH:-$root_dir/logs/qwen3vl_action_eval}"
qwen3vl_model_path="${QWEN3VL_MODEL_PATH:-/root/autodl-fs/models/Qwen3-VL-4B-Instruct}"
qwen3vl_adapter_path="${QWEN3VL_ADAPTER_PATH-/root/autodl-fs/checkpoints/uavon_qwen3vl_action_baseline}"

CUDA_VISIBLE_DEVICES="$cuda_visible_devices" python -u "$root_dir/src/eval_qwen3vl_action.py" \
    --name Qwen3VLAction \
    --maxActions "${UAV_ON_MAX_ACTIONS:-150}" \
    --eval_save_path "$eval_save_path" \
    --dataset_path "$dataset_path" \
    --is_fixed "${UAV_ON_IS_FIXED:-false}" \
    --xOy_step_size "${UAV_ON_XOY_STEP_SIZE:-5}" \
    --z_step_size "${UAV_ON_Z_STEP_SIZE:-2}" \
    --rotateAngle "${UAV_ON_ROTATE_ANGLE:-15}" \
    --gpu_id "$gpu_id" \
    --batchSize "${UAV_ON_BATCH_SIZE:-1}" \
    --simulator_tool_port "$sim_port" \
    --qwen3vl_model_path "$qwen3vl_model_path" \
    --qwen3vl_adapter_path "$qwen3vl_adapter_path" \
    --qwen3vl_max_new_tokens "${QWEN3VL_MAX_NEW_TOKENS:-8}" \
    --qwen3vl_eval_sample_mode "${QWEN3VL_EVAL_SAMPLE_MODE:-single}" \
    --qwen3vl_trajectory_max_steps "${QWEN3VL_TRAJECTORY_MAX_STEPS:-0}" \
    --qwen3vl_trajectory_kv_cache "${QWEN3VL_TRAJECTORY_KV_CACHE:-false}" \
    --qwen3vl_attn_implementation "${QWEN3VL_ATTN_IMPLEMENTATION:-flash_attention_2}"
