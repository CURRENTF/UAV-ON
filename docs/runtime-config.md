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
- `UAV_ON_XOY_STEP_SIZE`: default horizontal movement distance for action-only policies.
- `UAV_ON_Z_STEP_SIZE`: default vertical movement distance for action-only policies.
- `UAV_ON_ROTATE_ANGLE`: default yaw rotation angle for action-only policies.
- `UAV_ON_SCENE_BOOT_SECONDS`: scene boot wait after opening Unreal scenes.
- `UAV_ON_ACTION_TIMEOUT_SECONDS`: AirSim movement/rotation timeout.
- `UAV_ON_VERBOSE_POSE`: print pose-setting diagnostics.
- `UAV_ON_SET_POSE_SETTLE_SECONDS`: delay after kinematic pose updates.
- `UAV_ON_RGB_ONLY`: skip depth requests when only RGB is needed.
- `UAV_ON_FRONT_VIEW_ONLY`: request only AirSim camera `0` instead of all four
  UAV-ON cameras.
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

## Behavior Notes

`UAV_ON_RGB_ONLY=1` skips depth image requests. It is appropriate for the
current Qwen3-VL action datasets and policies because they consume RGB images
only.

`UAV_ON_FRONT_VIEW_ONLY=1` requests AirSim camera `0` only. It changes the
simulator image request rather than cropping a four-view observation after the
fact.

`UAV_ON_KINEMATIC_ACTIONS=1` uses `simSetVehiclePose` to place the UAV at the
next A* pose. This is fast and deterministic for offline expert data, but it
bypasses SimpleFlight movement dynamics and continuous path collision checks.
The implementation sets poses with `ignore_collision=True`, so collisions with
small or thin objects along the skipped path may not be detected. Images are
still real Unreal renders at the final pose, but the transition is not a fully
physical flight rollout. Use physics mode when the benchmark specifically needs
AirSim movement/collision fidelity.
