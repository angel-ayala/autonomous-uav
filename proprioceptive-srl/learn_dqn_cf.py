#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 12:45:59 2025

@author: angel
"""
import argparse
import numpy as np

from stable_baselines3 import DQN
from stable_baselines3.dqn.policies import DQNPolicy, CnnPolicy, MultiInputPolicy

from sb3_srl.dqn_srl import SRLDQNPolicy, SRLDQN

from utils import (
    DroneEnvMonitor,
    DroneExperimentCallback,
    args2ae_config,
    args2env_params,
    args2logpath,
    instance_env,
    parse_crazyflie_env_args,
    parse_memory_args,
    parse_srl_args,
    parse_training_args,
    parse_utils_args
)


def parse_agent_args(parser):
    arg_agent = parser.add_argument_group('Agent')
    arg_agent.add_argument("--lr", type=float, default=1e-4,
                           help='Critic function Adam learning rate.')
    arg_agent.add_argument("--tau", type=float, default=1.0,
                           help='Soft target update \tau.')
    arg_agent.add_argument("--discount-factor", type=float, default=0.99,
                           help='Discount factor \gamma.')
    arg_agent.add_argument("--train-freq", type=int, default=4,
                           help='Steps interval for policy optimization.')
    arg_agent.add_argument("--target-freq", type=int, default=10000,
                           help='Steps interval for target updates.')
    arg_agent.add_argument("--exploration-fraction", type=float, default=0.1,
                           help='fraction of entire training period over which the exploration rate is reduced.')
    arg_agent.add_argument("--exploration-init-eps", type=float, default=1.0,
                           help='initial value of random action probability.')
    arg_agent.add_argument("--exploration-final-eps", type=float, default=0.05,
                           help='final value of random action probability.')
    arg_agent.add_argument("--max-grad-norm", type=float, default=10,
                           help='The maximum value for the gradient clipping.')
    return arg_agent


def parse_args():
    # Argument parser
    parser = argparse.ArgumentParser()

    parse_crazyflie_env_args(parser)
    parse_agent_args(parser)
    parse_memory_args(parser)
    parse_srl_args(parser)
    parse_training_args(parser)
    parse_utils_args(parser)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    # th.manual_seed(args.seed)
    # np.random.seed(args.seed)
    env_params = args2env_params(args)

    # Environment
    environment_name = 'webots_drone:webots_drone/CrazyflieEnvDiscrete-v0'
    env = instance_env(environment_name, env_params, seed=args.seed)

    if args.is_srl:
        algo, policy = SRLDQN, SRLDQNPolicy
        # Autoencoder parameters
        ae_config = args2ae_config(args, env_params)
        # Policy args
        policy_args = {
            # 'net_arch': [256, 256],
            'ae_config': ae_config,
            'encoder_tau': args.encoder_tau
            }
    else:
        algo, policy = DQN, DQNPolicy
        policy_args = None
        if args.is_pixels:
            policy = CnnPolicy
            if args.is_vector:
                policy = MultiInputPolicy

    # Output log path
    log_path, exp_name, run_id = args2logpath(args, 'dqn')
    outpath = f"{log_path}/{exp_name}_{run_id+1}"

    # Experiment data log
    env = DroneEnvMonitor(env, store_path=f"{outpath}/history_training.csv", n_sensors=4)

    # Save a checkpoint every N steps
    agents_path = f"{outpath}/agents"
    experiment_callback = DroneExperimentCallback(
      env=env,
      save_freq=args.eval_interval,
      save_path=agents_path,
      name_prefix="rl_model",
      save_replay_buffer=False,
      save_vecnormalize=False,
      exp_args=args,
      out_path=f"{outpath}/arguments.json",
      memory_steps=args.memory_steps
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
                 target_update_interval=args.target_freq,
                 exploration_fraction=args.exploration_fraction,
                 exploration_initial_eps=args.exploration_init_eps,
                 exploration_final_eps=args.exploration_final_eps,
                 max_grad_norm=args.max_grad_norm,                 
                 stats_window_size=100,
                 tensorboard_log=log_path,
                 policy_kwargs=policy_args,
                 verbose=0,
                 seed=args.seed,
                 device="auto",
                 _init_setup_model=True)

    # Train the agent
    model.learn(total_timesteps=(args.steps + args.memory_steps),
                callback=experiment_callback,
                log_interval=4,
                progress_bar=True,
                tb_log_name=exp_name)
    model.save(f"{agents_path}/rl_model_final")
