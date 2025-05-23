"""
Serial sampler version of Atari DQN.  Increasing the number of parallel
environmnets (sampler batch_B) should improve the efficiency of the forward
pass for action sampling on the GPU.  Using a larger batch size in the algorithm
should improve the efficiency of the forward/backward passes during training.
(But both settings may impact hyperparameter selection and learning.)

"""
from rlpyt.envs.atari.atari_env import AtariTrajInfo
from rlpyt.utils.logging.context import logger_context

import wandb
import torch
import numpy as np

from src.models import SPRCatDqnModel
from src.rlpyt_utils import OneToOneSerialEvalCollector, SerialSampler, MinibatchRlEvalWandb
from src.algos import SPRCategoricalDQN
from src.agent import SPRAgent
from src.utils import set_config

from src.rlpyt_drone_env import (
    args2env_params,
    instance_env,
    wrap_env,
    DroneEnvMonitor,
    EnvWrapper,
    now_str
)


def build_and_train(game="pong", run_ID=0, cuda_idx=0, args=None):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    name = "dqn_" + game
    log_dir = "logs/CrazyflieEnvDiscrete-v0"

    # Environment
    environment_name = 'webots_drone:webots_drone/CrazyflieEnvDiscrete-v0'
    env_params = args2env_params(args)
    env = instance_env(environment_name, env_params, seed=args.seed)
    env = wrap_env(env, env_params)  # observation preprocesing
    monitor_env = DroneEnvMonitor(env, store_path=f"{log_dir}/run_{run_ID}/history_training.csv", n_sensors=4)

    config = set_config(args, game)

    sampler = SerialSampler(
        EnvCls=EnvWrapper,
        TrajInfoCls=AtariTrajInfo,  # default traj info + GameScore
        env_kwargs={'env': monitor_env},
        eval_env_kwargs={'env': monitor_env},
        batch_T=config['sampler']['batch_T'],
        batch_B=config['sampler']['batch_B'],
        max_decorrelation_steps=0,
        eval_CollectorCls=OneToOneSerialEvalCollector,
        eval_n_envs=0,
        eval_max_steps=60,
        eval_max_trajectories=1,
    )
    args.discount = config["algo"]["discount"]
    algo = SPRCategoricalDQN(optim_kwargs=config["optim"], jumps=args.jumps, **config["algo"])  # Run with defaults.
    agent = SPRAgent(ModelCls=SPRCatDqnModel, model_kwargs=config["model"], **config["agent"])

    wandb.config.update(config)
    runner = MinibatchRlEvalWandb(
        algo=algo,
        agent=agent,
        sampler=sampler,
        n_steps=args.n_steps,
        affinity=dict(cuda_idx=cuda_idx),
        log_interval_steps=args.n_steps//args.num_logs,
        seed=args.seed,
        final_eval_only=args.final_eval_only,
    )
    config = dict(game=game)

    with logger_context(log_dir, run_ID, name, config, snapshot_mode="gap", override_prefix=True, use_summary_writer=True):
        monitor_env.init_store()
        runner.train()

    quit()


def parse_args():
    import argparse
    from .run_utils import parse_crazyflie_env_args
    from .run_utils import parse_alg_args

    # Argument parser
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parse_alg_args(parser)
    parse_crazyflie_env_args(parser)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.public:
        wandb.init(anonymous="allow", config=args, tags=[args.tag] if args.tag else None, dir=args.wandb_dir)
    else:
        wandb.init(project=args.project, entity=args.entity, config=args, tags=[args.tag] if args.tag else None, dir=args.wandb_dir)
    wandb.config.update(vars(args))
    build_and_train(game=args.game,
                    run_ID=now_str(),
                    cuda_idx=args.cuda_idx,
                    args=args)
