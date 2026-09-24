#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 12:45:59 2025

@author: angel
"""
import argparse

from stable_baselines3 import TD3
from stable_baselines3.td3.policies import TD3Policy, CnnPolicy, MultiInputPolicy

from sb3_srl.td3_srl import SRLTD3Policy, SRLTD3

from sb3_srl.agent_utils import (
    args2srl_config,
    load_json_dict,
    parse_memory_args,
    parse_srl_args,
    parse_utils_args
)

from sb3_srl.proprioceptive_masks import mujoco_prop_mask

from utils.env_mujoco import (
    get_env,
    parse_mujoco_env_args,
    iterate_agents_evaluation
)


def parse_eval_args(parser):
    arg_eval = parser.add_argument_group('Evaluation')
    arg_eval.add_argument('--episode', type=int, default=-1,
                          help='Indicate the episode number to execute, set -1 for all of them')
    arg_eval.add_argument('--eval-steps', type=int, default=1000,
                          help='Number of evaluation steps.')
    arg_eval.add_argument('--eval-episodes', type=int, default=10,
                          help='Number of evaluation episodes.')
    arg_eval.add_argument('--record', action='store_true',
                          help='Specific if record or not a video simulation.')
    return arg_eval


def parse_args():
    # Argument parser
    parser = argparse.ArgumentParser()
    parse_memory_args(parser)
    parse_srl_args(parser)
    parse_eval_args(parser)
    parse_mujoco_env_args(parser)
    parse_utils_args(parser)
    return parser.parse_args()


if __name__ == '__main__':
    eval_args = parse_args()
    saved_args = load_json_dict(eval_args.logspath + '/arguments.json')
    # env_params = args2env_params(saved_args)

    # Environment
    env_id = saved_args['environment_id']
    render = None if not eval_args.record else 'rgb_array'
    env = get_env(env_id, seed=eval_args.seed, render_mode=render)
    env_params = {
        'action_shape': env.action_space.shape,
        'state_shape': env.observation_space.shape,
    }

    # Algorithm
    if saved_args['is_srl']:
        algo, policy = SRLTD3, SRLTD3Policy
        # Autoencoder parameters
        if saved_args['model_proprio']:
            env_params['prop_mask'] = mujoco_prop_mask(env_id)
        srl_config = args2srl_config(saved_args, env_params)
        # Policy args
        policy_args = {
            'net_arch': [saved_args['model_hidden_dim'], saved_args['model_hidden_dim']],
            'srl_config': srl_config
            }
    else:
        algo, policy = TD3, TD3Policy
        policy_args = None
        # if saved_args['is_pixels']:
        #     policy = CnnPolicy
        #     if saved_args['is_vector']:
        #         policy = MultiInputPolicy

    # Evaluation loop
    iterate_agents_evaluation(env, algo, eval_args)
