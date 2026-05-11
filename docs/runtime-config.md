# UAV-ON Runtime Config

Observed on 2026-05-11 in AutoDL under `/root/autodl-tmp/UAV-ON`.

## Purpose

`src/common/runtime_config.py` centralizes UAV-ON runtime options that can change
data generation or closed-loop evaluation behavior. The goal is to keep legacy
environment-variable compatibility while making resolved settings easy to record
in manifests.

## Runtime Behavior Variables

These variables are read through `UavOnRuntimeConfig`:

- `UAV_ON_INLINE_VECTOR_ENV`: use the in-process vector env formatter.
- `UAV_ON_VECTOR_ENV_START_METHOD`: multiprocessing start method for vector env workers.
- `UAV_ON_KINEMATIC_ACTIONS`: use `simSetVehiclePose` instead of AirSim async movement.
- `UAV_ON_SCENE_BOOT_SECONDS`: scene boot wait after opening Unreal scenes.
- `UAV_ON_ACTION_TIMEOUT_SECONDS`: AirSim movement/rotation timeout.
- `UAV_ON_VERBOSE_POSE`: print pose-setting diagnostics.
- `UAV_ON_SET_POSE_SETTLE_SECONDS`: delay after kinematic pose updates.
- `UAV_ON_RGB_ONLY`: skip depth requests when only RGB is needed.
- `UAV_ON_RGB_COMPRESS`: request compressed RGB images from AirSim.
- `UAV_ON_IMAGE_SETTLE_SECONDS`: delay before image capture.
- `UAV_ON_JPEG_OPTIMIZE`: enable Pillow JPEG optimization in Qwen3-VL data generation.
- `UAV_ON_ASTAR_PLAN_CACHE`: enable A* plan cache.

Framework variables such as `RANK`, `WORLD_SIZE`, `LOCAL_RANK`, `WANDB_*`,
`CUDA_VISIBLE_DEVICES`, and download mirrors such as `HF_ENDPOINT` remain outside
this helper because they are framework or shell-launch concerns.

## Manifest Outputs

Qwen3-VL action data generation writes the resolved config into
`export_stats.json` under `runtime_config`.

Qwen3-VL action eval writes:

- `runtime_config.launch.json`: settings resolved by the launch script before eval starts.
- `runtime_config.json`: settings resolved inside `src/eval_qwen3vl_action.py`.
- `pipeline_metadata.json`: final pipeline manifest, including runtime config when eval completes.

## Official Qwen3-VL DoRA Entry Points

Use tracked scripts for repeatable runs:

```bash
bash scripts/qwen3vl_dora_smoke_eval.sh
bash scripts/run_qwen3vl_dora_nonforward_1k_pipeline.sh
```

`scripts/tmp/` remains for local one-off debugging only.
