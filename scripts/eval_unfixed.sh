#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
root_dir=.
data_root="${UAV_ON_DATA_ROOT:-/data2/haojitai/datasets/uav-on}"
dataset_path="${UAV_ON_EVAL_DATASET:-$data_root/DATASET/merged_valset.json}"
cuda_visible_devices="${UAV_ON_CUDA_VISIBLE_DEVICES:-0}"
gpu_id="${UAV_ON_GPU_ID:-0}"
sim_port="${UAV_ON_SIM_PORT:-30000}"
echo "$PWD"

CUDA_VISIBLE_DEVICES="$cuda_visible_devices" python -u $root_dir/src/eval_2.py \
    --maxActions 150 \
    --eval_save_path $root_dir/unfixed_logs/scene \
    --dataset_path "$dataset_path" \
    --is_fixed false \
    --gpu_id "$gpu_id" \
    --batchSize 1 \
    --simulator_tool_port "$sim_port"
