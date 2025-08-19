#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 12:45:59 2025

@author: angel
"""
import argparse
import numpy as np
import gymnasium as gym

from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.td3.policies import TD3Policy, CnnPolicy, MultiInputPolicy
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

from sb3_srl.td3_srl import SRLTD3Policy, SRLTD3

from utils.agent import (
    args2ae_config,
    args2logpath,
    parse_memory_args,
    parse_srl_args,
    parse_utils_args
)
from utils.env_mujoco import get_env, parse_training_args


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
    arg_agent.add_argument("--policy-freq", type=int, default=2,
                           help='Steps interval for actor batch training.')
    arg_agent.add_argument("--policy-noise", type=float, default=0.2,
                           help='Policy noise value.')
    arg_agent.add_argument("--noise-clip", type=float, default=0.5,
                           help='Policy noise clip value.')
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
        algo, policy = SRLTD3, SRLTD3Policy
        # Autoencoder parameters
        ae_config = args2ae_config(args, env_params)
        # Policy args
        policy_args = {
            'net_arch': [256, 256],
            'ae_config': ae_config,
            'encoder_tau': args.encoder_tau
            }
    else:
        algo, policy = TD3, TD3Policy
        policy_args = None
        if args.is_pixels:
            policy = CnnPolicy
            if args.is_vector:
                policy = MultiInputPolicy

    # Output log path
    log_path, exp_name, run_id = args2logpath(args, 'td3', 'mujoco')
    outpath = f"{log_path}/{exp_name}_{run_id+1}"

    agents_path = f"{outpath}/agents"

    checkpoint_callback = CheckpointCallback(
        save_freq=args.eval_interval,
        save_path=agents_path,
        name_prefix="td3_mujoco_model",
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

    # Create action noise because TD3 and DDPG use a deterministic policy
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=args.exploration_noise * np.ones(n_actions))

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
                 action_noise=action_noise,
                 policy_delay=args.policy_freq,
                 target_policy_noise=args.policy_noise,
                 target_noise_clip=args.noise_clip,
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
