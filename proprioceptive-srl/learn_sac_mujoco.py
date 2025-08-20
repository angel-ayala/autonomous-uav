#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 12:45:59 2025

@author: angel
"""
import argparse
import numpy as np

from stable_baselines3 import SAC
from stable_baselines3.sac.policies import SACPolicy, CnnPolicy, MultiInputPolicy
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

from sb3_srl.sac_srl import SRLSACPolicy, SRLSAC

from utils.agent import (
    args2ae_config,
    parse_memory_args,
    parse_srl_args,
    parse_utils_args
)

from utils.env_mujoco import get_env, args2logpath, parse_training_args


def parse_agent_args(parser):
    arg_agent = parser.add_argument_group('Agent')
    arg_agent.add_argument("--lr", type=float, default=1e-3,
                           help='Critic function Adam learning rate.')
    arg_agent.add_argument("--tau", type=float, default=0.005,
                           help='Soft target update \tau.')
    arg_agent.add_argument("--exploration-noise", type=float, default=0.1,
                           help='Action noise during learning.')
    arg_agent.add_argument("--discount-factor", type=float, default=0.99,
                           help='Discount factor \gamma.')
    arg_agent.add_argument("--train-freq", type=int, default=1,
                           help='Steps interval for critic batch training.')
    arg_agent.add_argument("--target-update-freq", type=int, default=1,
                           help='Steps interval for target network update.')
    return arg_agent


def parse_args():
    # Argument parser
    parser = argparse.ArgumentParser()
    parse_agent_args(parser)
    parse_memory_args(parser)
    parse_srl_args(parser)
    parse_training_args(parser)
    parse_utils_args(parser)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    # Environment
    env = get_env(args.environment_id)

    env_params = {
        'action_shape': env.action_space.shape,
        'state_shape': env.observation_space.shape,
    }

    if args.is_srl:
        algo, policy = SRLSAC, SRLSACPolicy
        # Autoencoder parameters
        ae_config = args2ae_config(args, env_params)
        # Policy args
        policy_args = {
            'ae_config': ae_config,
            'encoder_tau': args.encoder_tau
            }
    else:
        algo, policy = SAC, SACPolicy
        policy_args = None
        # if args.is_pixels:
        #     policy = CnnPolicy
        #     if args.is_vector:
        #         policy = MultiInputPolicy

    # Output log path
    log_path, exp_name, run_id = args2logpath(args, 'sac', 'mujoco')
    outpath = f"{log_path}/{exp_name}_{run_id+1}"

    agents_path = f"{outpath}/agents"

    checkpoint_callback = CheckpointCallback(
        save_freq=args.eval_interval,
        save_path=agents_path,
        name_prefix="rl_model",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    evaluation = EvalCallback(
        get_env(args.environment_id),
        n_eval_episodes=args.eval_episodes,
        eval_freq=args.eval_interval,
        log_path=f"{outpath}/evaluations",
        best_model_save_path=agents_path,
        deterministic=True,
        render=False,
        verbose=1,
        warn=True,
        )

    # Create RL model
    model = algo(policy, env,
                 learning_rate=args.lr,
                 buffer_size=args.memory_capacity,  # 1e6
                 learning_starts=args.memory_steps,
                 batch_size=args.batch_size,
                 tau=args.tau,
                 gamma=args.discount_factor,
                 train_freq=args.train_freq,
                 gradient_steps=1,
                 action_noise=None,
                 ent_coef="auto",
                 target_update_interval=args.target_update_freq,
                 target_entropy="auto",
                 use_sde=False,
                 sde_sample_freq=-1,
                 use_sde_at_warmup=False,
                 stats_window_size=100,
                 tensorboard_log=log_path,
                 policy_kwargs=policy_args,
                 verbose=0,
                 seed=args.seed,
                 device="auto",
                 _init_setup_model=True)

    # Train the agent
    model.learn(total_timesteps=(args.steps + args.memory_steps),
                callback=[checkpoint_callback, evaluation],
                log_interval=4,
                progress_bar=True,
                tb_log_name=exp_name)

    model.save(f"{agents_path}/rl_model_final")
