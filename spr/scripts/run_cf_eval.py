#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 27 23:58:37 2025

@author: angel
"""
import argparse
import numpy as np
import torch

from src.rlpyt_drone_env import (
    args2env_params,
    iterate_agents_evaluation,
    instance_env,
    parse_alg_args,
    parse_crazyflie_env_args,
    wrap_env
)


def parse_eval_args(parser):
    arg_eval = parser.add_argument_group('Evaluation')
    arg_eval.add_argument('--episode', type=int, default=-1,
                          help='Indicate the episode number to execute, set -1 for all of them')
    arg_eval.add_argument('--eval-steps', type=int, default=60,
                          help='Number of evaluation steps.')
    arg_eval.add_argument('--eval-episodes', type=int, default=10,
                          help='Number of evaluation episodes.')
    arg_eval.add_argument('--logspath', type=str, default='logs/run_0', help='Specific output path of training results.')
    # arg_eval.add_argument('--record', action='store_true',
    #                       help='Specific if record or not a video simulation.')
    return arg_eval


def parse_args():
    # Argument parser
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parse_alg_args(parser)
    parse_crazyflie_env_args(parser)
    parse_eval_args(parser)
    return parser.parse_args()


args = parse_args()
np.random.seed(args.seed)
torch.manual_seed(args.seed)

# Environment
environment_name = 'webots_drone:webots_drone/CrazyflieEnvDiscrete-v0'
env_params = args2env_params(args)
env = instance_env(environment_name, env_params, seed=args.seed)
env = wrap_env(env, env_params)  # observation preprocesing

# Evaluation
iterate_agents_evaluation(env, args)
