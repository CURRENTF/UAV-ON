import os
import random
import time
from pathlib import Path
import sys

import numpy as np
import torch
import tqdm

sys.path.append(str(Path(str(os.getcwd())).resolve()))
from common.param import args
from common.runtime_config import write_runtime_config
from env_uav import AirVLNENV
from src.closeloop_util import BatchIterator, EvalBatchState, initialize_env_eval
from utils.logger import logger
from model_wrapper.Qwen3VLAction import Qwen3VLAction


def eval(model_wrapper: Qwen3VLAction, env: AirVLNENV, is_fixed, save_eval_path):
    with torch.no_grad():
        data = BatchIterator(env)
        data_len = len(data)
        pbar = tqdm.tqdm(total=data_len, desc="batch")
        cnt = 0
        while True:
            env_batch = env.next_minibatch(skip_scenes=[])
            if env_batch is None:
                break

            batch_state = EvalBatchState(
                batch_size=env.batch_size,
                env_batchs=env_batch,
                env=env,
                save_eval_path=save_eval_path,
            )

            pbar.update(n=env.batch_size)
            inputs, user_prompts = model_wrapper.prepare_inputs(batch_state.episodes, is_fixed)
            cnt += env.batch_size
            for t in range(args.maxActions):
                logger.info("Step: {} \t Completed: {} / {}".format(t, cnt - batch_state.skips.count(False), data_len))

                start1 = time.time()
                actions, steps_size, dones = model_wrapper.run(inputs, is_fixed)
                print("get actions time:", time.time() - start1)

                for i in range(env.batch_size):
                    if dones[i]:
                        batch_state.dones[i] = True

                for i in range(len(actions)):
                    print(actions[i], ":", steps_size[i])

                env.makeActions(actions, steps_size, is_fixed)
                obs = env.get_obs()
                batch_state.update_from_env_output(obs, user_prompts, actions, steps_size, is_fixed)
                batch_state.update_metric()
                if batch_state.check_batch_termination(t):
                    break

                inputs, user_prompts = model_wrapper.prepare_inputs(batch_state.episodes, is_fixed)
        try:
            pbar.close()
        except Exception:
            pass


if __name__ == "__main__":
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    env = initialize_env_eval(dataset_path=args.dataset_path, save_path=args.eval_save_path)
    fixed = args.is_fixed

    save_eval_path = os.path.join(args.eval_save_path, args.name)
    if not os.path.exists(args.eval_save_path):
        os.makedirs(args.eval_save_path)
    write_runtime_config(
        Path(args.eval_save_path) / "runtime_config.json",
        scene_boot_seconds_default=75.0,
        extra={
            "eval": {
                "name": args.name,
                "dataset_path": args.dataset_path,
                "eval_save_path": args.eval_save_path,
                "max_actions": args.maxActions,
                "xOy_step_size": args.xOy_step_size,
                "z_step_size": args.z_step_size,
                "rotateAngle": args.rotateAngle,
                "batch_size": args.batchSize,
                "is_fixed": args.is_fixed,
                "simulator_tool_port": args.simulator_tool_port,
                "qwen3vl_eval_sample_mode": args.qwen3vl_eval_sample_mode,
                "qwen3vl_trajectory_max_steps": args.qwen3vl_trajectory_max_steps,
                "qwen3vl_trajectory_kv_cache": args.qwen3vl_trajectory_kv_cache,
                "qwen3vl_max_new_tokens": args.qwen3vl_max_new_tokens,
            }
        },
    )

    model_wrapper = Qwen3VLAction(fixed=fixed, batch_size=args.batchSize)
    eval(model_wrapper=model_wrapper, env=env, is_fixed=fixed, save_eval_path=save_eval_path)

    env.delete_VectorEnvUtil()
