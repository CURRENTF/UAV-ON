#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

run_id="${UAV_ON_RUN_ID:-front_rgb_3m_20k_balanced_fast_approx_complete_traj}"
log_dir="${UAV_ON_RUN_LOG_DIR:-/root/autodl-fs/logs}"
mkdir -p "$log_dir"

export PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/uavon/bin/python}"
export UAV_ON_DATA_ROOT="${UAV_ON_DATA_ROOT:-/root/autodl-fs/datasets/UAV-ON-train-unpacked}"
export UAV_ON_SERVER_GPUS="${UAV_ON_SERVER_GPUS:-0}"
export UAV_ON_SIM_PORT="${UAV_ON_SIM_PORT:-30000}"
export UAV_ON_CUDA_VISIBLE_DEVICES="${UAV_ON_CUDA_VISIBLE_DEVICES:-0}"
export UAV_ON_GPU_ID="${UAV_ON_GPU_ID:-0}"

export UAV_ON_SCENE_SHARD_SOURCE="${UAV_ON_SCENE_SHARD_SOURCE:-/root/autodl-fs/datasets/uavon_train_generation_balanced.json}"
export UAV_ON_SCENE_SHARD_MANIFEST="${UAV_ON_SCENE_SHARD_MANIFEST:-/root/autodl-fs/datasets/uavon_train_generation_scene_shards/manifest.json}"
export UAV_ON_GENERATE_BASE_OUTPUT="${UAV_ON_GENERATE_BASE_OUTPUT:-/root/autodl-fs/datasets/uavon_qwen3vl_action_front_rgb_3m_20k_balanced_fast_approx_complete_traj}"
export UAV_ON_GENERATE_SAMPLES_PER_SCENE="${UAV_ON_GENERATE_SAMPLES_PER_SCENE:-2000}"
export UAV_ON_GENERATE_COMPLETE_TRAJECTORIES="${UAV_ON_GENERATE_COMPLETE_TRAJECTORIES:-1}"
export UAV_ON_COLLECTION_TARGET_TOTAL_SAMPLES="${UAV_ON_COLLECTION_TARGET_TOTAL_SAMPLES:-20000}"
export UAV_ON_COLLECTION_TARGET_SAMPLES_PER_SCENE="${UAV_ON_COLLECTION_TARGET_SAMPLES_PER_SCENE:-2000}"
export UAV_ON_COLLECTION_SCENE_BALANCE_MODE="${UAV_ON_COLLECTION_SCENE_BALANCE_MODE:-per_scene_shards_2000_each_complete_trajectories}"

export UAV_ON_BATCH_SIZE="${UAV_ON_BATCH_SIZE:-4}"
export UAV_ON_VECTOR_ENV_START_METHOD="${UAV_ON_VECTOR_ENV_START_METHOD:-fork}"
export UAV_ON_SCENE_BOOT_SECONDS="${UAV_ON_SCENE_BOOT_SECONDS:-75}"
export UAV_ON_KINEMATIC_ACTIONS="${UAV_ON_KINEMATIC_ACTIONS:-1}"
export UAV_ON_ACTION_TIMEOUT_SECONDS="${UAV_ON_ACTION_TIMEOUT_SECONDS:-12}"
export UAV_ON_RESET_RETRY_ATTEMPTS="${UAV_ON_RESET_RETRY_ATTEMPTS:-4}"
export UAV_ON_RESET_RETRY_SLEEP_SECONDS="${UAV_ON_RESET_RETRY_SLEEP_SECONDS:-20}"

export UAV_ON_RGB_ONLY="${UAV_ON_RGB_ONLY:-1}"
export UAV_ON_FRONT_VIEW_ONLY="${UAV_ON_FRONT_VIEW_ONLY:-1}"
export UAV_ON_RGB_COMPRESS="${UAV_ON_RGB_COMPRESS:-0}"
export UAV_ON_JPEG_OPTIMIZE="${UAV_ON_JPEG_OPTIMIZE:-0}"
export UAV_ON_INLINE_VECTOR_ENV="${UAV_ON_INLINE_VECTOR_ENV:-1}"
export UAV_ON_IMAGE_SETTLE_SECONDS="${UAV_ON_IMAGE_SETTLE_SECONDS:-0}"
export UAV_ON_SET_POSE_SETTLE_SECONDS="${UAV_ON_SET_POSE_SETTLE_SECONDS:-0}"
export UAV_ON_IMAGE_VERIFY_TIMEOUT_SECONDS="${UAV_ON_IMAGE_VERIFY_TIMEOUT_SECONDS:-1}"
export UAV_ON_SET_POSE_VERIFY_TIMEOUT_SECONDS="${UAV_ON_SET_POSE_VERIFY_TIMEOUT_SECONDS:-1}"

export UAV_ON_ASTAR_PLAN_CACHE="${UAV_ON_ASTAR_PLAN_CACHE:-1}"
export UAV_ON_ASTAR_VOXEL_RESOLUTION="${UAV_ON_ASTAR_VOXEL_RESOLUTION:-3.0}"
export UAV_ON_ASTAR_MAX_MOVE_VOXELS="${UAV_ON_ASTAR_MAX_MOVE_VOXELS:-1}"
export UAV_ON_ASTAR_TURN_COOLDOWN_AFTER="${UAV_ON_ASTAR_TURN_COOLDOWN_AFTER:-0}"
export UAV_ON_ASTAR_TURN_COOLDOWN_STEPS="${UAV_ON_ASTAR_TURN_COOLDOWN_STEPS:-0}"

export UAV_ON_GENERATE_FLUSH_EVERY="${UAV_ON_GENERATE_FLUSH_EVERY:-50}"
export UAV_ON_GENERATE_STATUS_EVERY="${UAV_ON_GENERATE_STATUS_EVERY:-50}"
export UAV_ON_COLLECTION_VARIANT="${UAV_ON_COLLECTION_VARIANT:-front_rgb_3m_20k_balanced_fast_approx_complete_traj}"
export UAV_ON_COLLECTION_TAGS="${UAV_ON_COLLECTION_TAGS:-3m,front-only,rgb-only,20k,balanced-scenes,fast,approx,non-strict,complete-trajectories}"
export UAV_ON_COLLECTION_NOTE="${UAV_ON_COLLECTION_NOTE:-Movement labels use approximate/non-strict 3m A* units via astar_voxel_resolution=3.0 and astar_max_move_voxels=1; rotation labels remain degrees. complete_trajectories=1 means per-scene sample targets are lower bounds and the active batch is allowed to finish at stop.}"

server_log="$log_dir/${run_id}_server.log"
generator_log="$log_dir/${run_id}_generator.log"
server_pid_file="$log_dir/${run_id}_server.pid"
generator_pid_file="$log_dir/${run_id}_generator.pid"

if pgrep -af "AirVLNSimulatorServerTool.py --port ${UAV_ON_SIM_PORT}" >/dev/null; then
  echo "[run] reusing existing simulator server on port ${UAV_ON_SIM_PORT}" | tee -a "$server_log"
else
  echo "[run] starting simulator server on port ${UAV_ON_SIM_PORT}" | tee -a "$server_log"
  setsid bash -lc 'cd /root/autodl-tmp/UAV-ON && bash scripts/start_server_train.sh' \
    >> "$server_log" 2>&1 < /dev/null &
  echo "$!" > "$server_pid_file"
fi

echo "[run] output_root=${UAV_ON_GENERATE_BASE_OUTPUT}" | tee -a "$generator_log"
echo "[run] complete_trajectories=${UAV_ON_GENERATE_COMPLETE_TRAJECTORIES}" | tee -a "$generator_log"
echo "[run] starting generator $(date -Is)" | tee -a "$generator_log"
setsid bash -lc 'cd /root/autodl-tmp/UAV-ON && bash scripts/generate_qwen3vl_action_dataset_all_train_scenes.sh' \
  >> "$generator_log" 2>&1 < /dev/null &
echo "$!" > "$generator_pid_file"

echo "[run] generator_pid=$(cat "$generator_pid_file")"
echo "[run] generator_log=$generator_log"
echo "[run] server_log=$server_log"
echo "[run] output_root=${UAV_ON_GENERATE_BASE_OUTPUT}"
