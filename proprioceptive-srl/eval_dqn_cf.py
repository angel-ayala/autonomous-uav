#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 12:45:59 2025

@author: angel
"""
import argparse

from stable_baselines3 import DQN
from stable_baselines3.dqn.policies import DQNPolicy, CnnPolicy, MultiInputPolicy

from sb3_srl.dqn_srl import SRLDQNPolicy, SRLDQN

from utils.agent import (
    args2ae_config,
    load_json_dict,
    parse_memory_args,
    parse_srl_args,
    parse_utils_args
)

from utils.env_drone import (
    args2env_params,
    instance_env,
    parse_crazyflie_env_args,
    iterate_agents_evaluation
)


def parse_eval_args(parser):
    arg_eval = parser.add_argument_group('Evaluation')
    arg_eval.add_argument('--episode', type=int, default=-1,
                          help='Indicate the episode number to execute, set -1 for all of them')
    arg_eval.add_argument('--eval-steps', type=int, default=60,
                          help='Number of evaluation steps.')
    arg_eval.add_argument('--eval-episodes', type=int, default=10,
                          help='Number of evaluation episodes.')
    # arg_eval.add_argument('--record', action='store_true',
    #                       help='Specific if record or not a video simulation.')
    return arg_eval


def parse_args():
    # Argument parser
    parser = argparse.ArgumentParser()
    parse_crazyflie_env_args(parser)
    parse_memory_args(parser)
    parse_srl_args(parser)
    parse_eval_args(parser)
    parse_utils_args(parser)
    return parser.parse_args()


if __name__ == '__main__':
    eval_args = parse_args()
    saved_args = load_json_dict(eval_args.logspath + '/arguments.json')
    env_params = args2env_params(saved_args)

    # Environment
    environment_name = 'webots_drone:webots_drone/CrazyflieEnvDiscrete-v0'
    env = instance_env(environment_name, env_params, seed=eval_args.seed)

    # Algorithm
    if saved_args['is_srl']:
        algo, policy = SRLDQN, SRLDQNPolicy
        # Autoencoder parameters
        ae_config = args2ae_config(saved_args, env_params)
        # Policy args
        policy_args = {
            #'net_arch': [256, 256],
            'ae_config': ae_config,
            'encoder_tau': saved_args['encoder_tau']
            }
    else:
        algo, policy = DQN, DQNPolicy
        policy_args = None
        if saved_args['is_pixels']:
            policy = CnnPolicy
            if saved_args['is_vector']:
                policy = MultiInputPolicy

    # Evaluation loop
    iterate_agents_evaluation(env, algo, eval_args)
