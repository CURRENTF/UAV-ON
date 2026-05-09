#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
root_dir=.
lightning_root="${LIGHTNING_VLN_ROOT:-/root/autodl-tmp/LightningVLN}"
data_root="${UAV_ON_DATA_ROOT:-/data2/haojitai/datasets/uav-on}"
dataset_path="${UAV_ON_EVAL_DATASET:-$data_root/DATASET/merged_valset.json}"
cuda_visible_devices="${UAV_ON_CUDA_VISIBLE_DEVICES:-0}"
gpu_id="${UAV_ON_GPU_ID:-0}"
sim_port="${UAV_ON_SIM_PORT:-30000}"
eval_save_path="${UAV_ON_EVAL_SAVE_PATH:-$root_dir/logs/unix_action_eval}"
unix_model_path="${UNIX_MODEL_PATH:-/root/autodl-fs/models/Uni-X-3B}"
unix_adapter_path="${UNIX_ADAPTER_PATH:-/root/autodl-fs/checkpoints/uavon_astar_action_baseline}"
unix_vqgan_dir="${UNIX_VQGAN_DIR:-/root/autodl-fs/models/chameleon_vqgan_tokenizer/tokenizer}"

export LIGHTNING_VLN_ROOT="$lightning_root"
export PYTHONPATH="${PYTHONPATH:-}:$lightning_root/src:$root_dir"

CUDA_VISIBLE_DEVICES="$cuda_visible_devices" python -u "$root_dir/src/eval_unix_action.py" \
    --name UniXAction \
    --maxActions "${UAV_ON_MAX_ACTIONS:-150}" \
    --eval_save_path "$eval_save_path" \
    --dataset_path "$dataset_path" \
    --is_fixed "${UAV_ON_IS_FIXED:-false}" \
    --gpu_id "$gpu_id" \
    --batchSize "${UAV_ON_BATCH_SIZE:-1}" \
    --simulator_tool_port "$sim_port" \
    --unix_model_path "$unix_model_path" \
    --unix_adapter_path "$unix_adapter_path" \
    --unix_vqgan_dir "$unix_vqgan_dir" \
    --unix_repo_path "$lightning_root/src/lightning_vln/modeling/uni_x" \
    --unix_max_new_tokens "${UNIX_MAX_NEW_TOKENS:-8}" \
    --unix_vq_batch_size "${UNIX_VQ_BATCH_SIZE:-1}"
