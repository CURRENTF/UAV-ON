import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
import tqdm

sys.path.append(str(Path(str(os.getcwd())).resolve()))
from common.param import args
from env_uav import AirVLNENV
from model_wrapper.AStarOracle import AStarOracle
from model_wrapper.base_model import BaseModelWrapper
from src.closeloop_util import BatchIterator, EvalBatchState, initialize_env_eval
from utils.logger import logger


def eval(modelWrapper: BaseModelWrapper, env: AirVLNENV, is_fixed, save_eval_path):
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
            modelWrapper.prepare_batch(env=env, batch=env_batch)

            pbar.update(n=env.batch_size)
            cnt += env.batch_size
            inputs, user_prompts = modelWrapper.prepare_inputs(batch_state.episodes, is_fixed)

            for t in range(args.maxActions):
                logger.info("Step: {} \t Completed: {} / {}".format(t, cnt - batch_state.skips.count(False), data_len))

                start_time = time.time()
                actions, steps_size, dones = modelWrapper.run(inputs, is_fixed)
                print("get actions time:", time.time() - start_time)

                for i in range(env.batch_size):
                    if dones[i]:
                        batch_state.dones[i] = True

                for i in range(len(actions)):
                    print(actions[i], ":", steps_size[i])

                env.makeActions(actions, steps_size, is_fixed)
                obs = env.get_obs()
                batch_state.update_from_env_output(obs, user_prompts, actions, steps_size, is_fixed)
                batch_state.update_metric()

                is_terminate = batch_state.check_batch_termination(t)
                if is_terminate:
                    break

                inputs, user_prompts = modelWrapper.prepare_inputs(batch_state.episodes, is_fixed)

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

    if args.is_fixed:
        raise RuntimeError("AStarOracle requires --is_fixed false so 1-unit action step sizes are honored.")

    env = initialize_env_eval(dataset_path=args.dataset_path, save_path=args.eval_save_path)
    save_eval_path = os.path.join(args.eval_save_path, args.name)
    if not os.path.exists(args.eval_save_path):
        os.makedirs(args.eval_save_path)

    modelWrapper = AStarOracle(batch_size=args.batchSize, args=args)
    eval(modelWrapper=modelWrapper, env=env, is_fixed=args.is_fixed, save_eval_path=save_eval_path)

    env.delete_VectorEnvUtil()
