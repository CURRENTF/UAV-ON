#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

data_root="${UAV_ON_DATA_ROOT:-/data2/haojitai/datasets/uav-on}"
sim_port="${UAV_ON_SIM_PORT:-30000}"
server_gpus="${UAV_ON_SERVER_GPUS:-1,2,3,4}"
python_bin="${PYTHON_BIN:-/root/miniconda3/envs/uavon/bin/python}"

"$python_bin" airsim_plugin/AirVLNSimulatorServerTool.py \
  --port "$sim_port" \
  --gpus "$server_gpus" \
  --root_path "$data_root/TEST_ENVS"
