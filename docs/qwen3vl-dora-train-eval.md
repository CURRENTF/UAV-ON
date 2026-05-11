# Qwen3-VL DoRA Train/Eval Pipeline

Observed on 2026-05-11 in AutoDL under `/root/autodl-tmp/UAV-ON`.

## Purpose

Run the Qwen3-VL UAV-ON action policy with a reproducible shell pipeline:

1. Smoke-test closed-loop Qwen3-VL + PEFT adapter evaluation.
2. Rebalance the 20k A* action JSONL toward non-`forward` actions.
3. Train a DoRA adapter for 1000 steps.
4. Evaluate the trained adapter on UAV-ON.

## Scripts

- Smoke eval:
  `/root/autodl-tmp/UAV-ON/scripts/qwen3vl_dora_smoke_eval.sh`
- Full pipeline:
  `/root/autodl-tmp/UAV-ON/scripts/run_qwen3vl_dora_nonforward_1k_pipeline.sh`
- Dataset rebalance helper:
  `/root/autodl-tmp/LightningVLN/scripts/make_qwen3vl_action_rebalanced_dataset.py`
- Qwen3-VL training wrapper:
  `/root/autodl-tmp/LightningVLN/scripts/train_qwen3vl_action_baseline.sh`

`scripts/tmp` is gitignored and should be used only for local one-off debugging.

## Training Settings

The full pipeline passes these settings to LightningVLN:

- `PEFT_METHOD=dora`
- `PER_DEVICE_TRAIN_BATCH_SIZE=4`
- `GRADIENT_ACCUMULATION_STEPS=1`
- `LEARNING_RATE=1e-5`
- `LR_SCHEDULER_TYPE=linear`
- `MAX_STEPS=1000`
- `SAVE_STEPS=200`
- `SAVE_TOTAL_LIMIT=5`
- `REPORT_TO=wandb` by default; set `REPORT_TO=none` explicitly to disable it

## Action Schema

Qwen3-VL action prompts and parsers now use prompt/schema version
`qwen3vl_action_v1`.

The nominal UAV-ON action set is:

- `forward`
- `left`
- `right`
- `rotl`
- `rotr`
- `ascend`
- `descend`
- `stop`

The current A* oracle can execute in the UAV-ON environment but emits labels by
rotating toward a direction and then moving forward. Its label set is therefore:

- `forward`
- `rotl`
- `rotr`
- `ascend`
- `descend`
- `stop`

`left` and `right` are valid executable actions and remain in the model-facing
allowed action list, but they are not present in the current A* SFT labels unless
the data generator or oracle is changed.

## Data Rebalancing

Source:

`/root/autodl-fs/datasets/uavon_qwen3vl_action_train_astar_20k/train.jsonl`

Output:

`/root/autodl-fs/datasets/uavon_qwen3vl_action_train_astar_20k_nonforward80_seed42/train.jsonl`

The default rebalance keeps all non-`forward` examples and samples enough
`forward` examples to target a 20% final `forward` fraction. The observed output
distribution was:

- `ascend`: 9
- `descend`: 1889
- `forward`: 2135
- `rotl`: 3198
- `rotr`: 3193
- `stop`: 253

## Run

Smoke eval only:

```bash
cd /root/autodl-tmp/UAV-ON
SMOKE_TASKS=1 UAV_ON_MAX_ACTIONS=1 \
UAV_ON_EVAL_SAVE_PATH=/root/autodl-fs/evals/qwen3vl_dora_smoke_eval_current \
bash scripts/qwen3vl_dora_smoke_eval.sh
```

Full train and eval pipeline:

```bash
cd /root/autodl-tmp/UAV-ON
RUN_SMOKE=0 REBALANCE_OVERWRITE=1 \
bash scripts/run_qwen3vl_dora_nonforward_1k_pipeline.sh
```

Detached launch:

```bash
cd /root/autodl-tmp/UAV-ON
setsid bash -lc 'cd /root/autodl-tmp/UAV-ON && RUN_SMOKE=0 REBALANCE_OVERWRITE=1 bash scripts/run_qwen3vl_dora_nonforward_1k_pipeline.sh' \
  >> /root/autodl-fs/logs/qwen3vl_nonforward80_dora_1k_pipeline_launcher.log 2>&1 < /dev/null &
```

## Outputs

- Rebalanced stats:
  `/root/autodl-fs/datasets/uavon_qwen3vl_action_train_astar_20k_nonforward80_seed42/rebalance_stats.json`
- Checkpoints:
  `/root/autodl-fs/checkpoints/uavon_qwen3vl_action_astar_20k_nonforward80_dora_bs4_lr1e-5_linear_1k`
- Final eval:
  `/root/autodl-fs/evals/uavon_qwen3vl_action_nonforward80_dora_bs4_lr1e-5_linear_1k_citypark_slum_20`
- Pipeline log:
  `/root/autodl-fs/logs/qwen3vl_nonforward80_dora_1k_pipeline_orchestrator.log`
- Train log:
  `/root/autodl-fs/logs/qwen3vl_nonforward80_dora_1k_train.log`
- Eval log:
  `/root/autodl-fs/logs/qwen3vl_nonforward80_dora_1k_eval.log`
- Summary log:
  `/root/autodl-fs/logs/qwen3vl_nonforward80_dora_1k_summary.log`
- Runtime config:
  `<eval_root>/runtime_config.launch.json` and `<eval_root>/runtime_config.json`
- Pipeline metadata:
  `<eval_root>/pipeline_metadata.json`

## Notes

The smoke eval uses the `kv` conda environment because it has the Qwen3-VL
runtime dependencies. On this machine, `yacs==0.1.8` and `numba==0.65.1` were
added to `kv` after the first smoke attempt exposed missing imports from the
UAV-ON environment code.

Runtime behavior variables are centralized in `src/common/runtime_config.py`.
See `docs/runtime-config.md` for the list of `UAV_ON_*` variables that affect
data generation and closed-loop evaluation.
