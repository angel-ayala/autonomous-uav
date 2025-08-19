#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 12:48:04 2025

@author: angel
"""
from typing import Callable
import time
import json
import sys
import numpy as np
import gymnasium as gym
from pathlib import Path
from natsort import natsorted

from stable_baselines3.common.utils import get_latest_run_id


def parse_memory_args(parser):
    arg_mem = parser.add_argument_group('Memory buffer')
    arg_mem.add_argument("--memory-capacity", type=int, default=65536,  # 2**16
                           help='Maximum number of transitions in the Experience replay buffer.')
    arg_mem.add_argument("--memory-prioritized", action='store_true',
                           help='Whether if memory buffer is Prioritized experiencie replay or not.')
    arg_mem.add_argument("--prioritized-alpha", type=float, default=0.6,
                           help='Alpha prioritization exponent for PER.')
    arg_mem.add_argument("--prioritized-initial-beta", type=float, default=0.4,
                           help='Beta bias for sampling for PER.')
    arg_mem.add_argument("--beta-steps", type=float, default=112500,
                           help='Beta bias steps to reach 1.')
    return arg_mem


def parse_srl_args(parser):
    arg_srl = parser.add_argument_group(
        'State representation learning variation')
    arg_srl.add_argument("--is-srl", action='store_true',
                         help='Whether if method is SRL-based or not.')
    arg_srl.add_argument("--latent-dim", type=int, default=32,
                         help='Number of features in the latent representation Z.')
    arg_srl.add_argument("--hidden-dim", type=int, default=512,
                         help='Number of units in the hidden layers.')
    arg_srl.add_argument("--num-filters", type=int, default=32,
                         help='Number of filters in the CNN hidden layers.')
    arg_srl.add_argument("--num-layers", type=int, default=1,
                         help='Number of hidden layers.')
    arg_srl.add_argument("--encoder-lr", type=float, default=1e-3,
                         help='Encoder function Adam learning rate.')
    arg_srl.add_argument("--encoder-tau", type=float, default=0.999,
                         help='Encoder tau polyak update.')
    arg_srl.add_argument("--encoder-steps", type=int, default=9000,
                         help='Steps of no improvement to stop Encoder gradient.')
    arg_srl.add_argument("--decoder-lr", type=float, default=1e-3,
                         help='Decoder function Adam learning rate.')
    arg_srl.add_argument("--decoder-latent-lambda", type=float, default=1e-6,
                         help='Decoder regularization lambda value.')
    arg_srl.add_argument("--decoder-weight-decay", type=float, default=1e-7,
                         help='Decoder function Adam weight decay value.')
    arg_srl.add_argument("--representation-freq", type=int, default=1,
                         help='Steps interval for AE batch training.')
    arg_srl.add_argument("--encoder-only", action='store_true',
                         help='Whether if use the SRL loss.')
    arg_srl.add_argument("--model-reconstruction", action='store_true',
                         help='Whether if use the Reconstruction model.')
    arg_srl.add_argument("--model-spr", action='store_true',
                         help='Whether if use the SelfPredictive model.')
    arg_srl.add_argument("--model-reconstruction-dist", action='store_true',
                         help='Whether if use the ReconstructionDist reconstruction model.')
    arg_srl.add_argument("--model-ispr", action='store_true',
                         help='Whether if use the InfoNCE SimpleSPR version model.')
    arg_srl.add_argument("--model-i2spr", action='store_true',
                         help='Whether if use the Introspective InfoNCE SimpleSPR model.')
    arg_srl.add_argument("--introspection-lambda", type=float, default=0,
                         help='Introspection loss function lambda value, >0 to use introspection.')
    arg_srl.add_argument("--joint-optimization", action='store_true',
                         help='Whether if jointly optimize representation with RL updates.')
    arg_srl.add_argument("--model-ispr-mumo", action='store_true',
                         help='Whether if use the InfoNCE SimpleSPR Multimodal version model.')
    arg_srl.add_argument("--model-proprio", action='store_true',
                         help='Whether if use the Proprioceptive version model.')
    arg_srl.add_argument("--use-stochastic", action='store_true',
                         help='Whether if use the Stochastic version model.')
    return arg_srl


def parse_training_args(parser):
    arg_training = parser.add_argument_group('Training')
    arg_training.add_argument("--steps", type=int, default=450000,  # 25h at 25 frames
                              help='Number of training steps.')
    arg_training.add_argument('--memory-steps', type=int, default=2048,
                              help='Number of steps for initial population of the Experience replay buffer.')
    arg_training.add_argument("--batch-size", type=int, default=128,
                              help='Minibatch size for training.')
    arg_training.add_argument('--eval-interval', type=int, default=9000,  # 30m at 25 frames
                              help='Steps interval for progress evaluation.')
    arg_training.add_argument('--eval-steps', type=int, default=60,  # 1m at 25 frames
                              help='Number of evaluation steps.')
    return arg_training


def parse_utils_args(parser):
    arg_utils = parser.add_argument_group('Utils')
    arg_utils.add_argument('--use-cuda', action='store_true',
                           help='Flag specifying whether to use the GPU.')
    arg_utils.add_argument('--seed', type=int, default=666,
                           help='Seed valu for torch and nummpy.')
    arg_utils.add_argument('--logspath', type=str, default=None,
                           help='Specific output path for training results.')
    return arg_utils


def args2ae_config(args, env_params):
    _args = args
    if not isinstance(_args, dict):
        _args = vars(_args)
    model_params = {
        'action_shape': env_params['action_shape'],
        'state_shape': env_params['state_shape'],
        'latent_dim': _args.get('latent_dim', 32),
        'layers_dim': [_args.get('hidden_dim', 256)] * _args.get('num_layers', 2),
        'layers_filter': [_args.get('num_filters', 32)] * _args.get('num_layers', 2),
        'encoder_lr': _args.get('encoder_lr', 1e-3),
        'decoder_lr': _args.get('decoder_lr', 1e-3),
        'encoder_only': _args.get('encoder_only', False),
        'encoder_steps': _args.get('encoder_steps', 9000),
        'decoder_lambda': _args.get('decoder_lambda', 1e-6),
        'decoder_weight_decay': _args.get('decoder_weight_decay', 1e-7),
        'joint_optimization': _args.get('joint_optimization', False),
        'introspection_lambda': _args.get('introspection_lambda', 0.),
        'is_pixels': _args.get('is_pixels', False),
        'is_multimodal': _args.get('is_pixels', False) and _args.get('is_vector', False),
        }

    if _args.get('model_reconstruction', False):
        model_name = 'Reconstruction'
    elif _args.get('model_spr', False):
        model_name = 'SelfPredictive'
    elif _args.get('model_reconstruction_dist', False):
        model_name = 'ReconstructionDist'
    elif _args.get('model_ispr', False):
        model_name = 'InfoSPR'
    elif _args.get('model_i2spr', False):
        model_name = 'IntrospectiveInfoSPR'
    elif _args.get('model_proprio', False):
        model_name = 'Proprioceptive'
    else:
        raise ValueError('SRL model not recognized...')
    
    if _args.get('use_stochastic', False):
        model_name += 'Stochastic'

    return model_name, model_params


def args2logpath(args, algo, env_uav='cf'):
    if args.logspath is None:
        if args.is_pixels and args.is_vector:
            path_prefix = 'multi'
        else:
            path_prefix = 'pixel' if args.is_pixels else 'vector'
        # Summary folder
        outfolder = f"logs_{env_uav}_{path_prefix}"
    else:
        outfolder = args.logspath

    path_suffix = ''
    # method labels
    if args.model_reconstruction:
        path_suffix += '-rec'
    if args.model_spr:
        path_suffix += '-spr'
    if args.model_reconstruction_dist:
        path_suffix += '-drec'
    if args.model_ispr:
        path_suffix += '-ispr'
    if args.model_i2spr:
        path_suffix += '-i2spr'
    if args.model_ispr_mumo:
        path_suffix += '-ispr-custom'
    if args.model_proprio:
        path_suffix += '-proprio'
    # extra labels
    if args.introspection_lambda != 0.:
        path_suffix += '-intr'
    if args.joint_optimization:
        path_suffix += '-joint'
    if args.use_stochastic:
        path_suffix += '-stch'
    exp_name = f"{algo}{path_suffix}"

    latest_run_id = get_latest_run_id(outfolder, exp_name)

    return outfolder, exp_name, latest_run_id


def save_dict_json(dict2save, json_path):
    proc_dic = dict2save.copy()
    dict_json = json.dumps(proc_dic,
                           indent=4,
                           default=lambda o: str(o))
    with open(json_path, 'w') as jfile:
        jfile.write(dict_json)
    return dict_json


def load_json_dict(json_path):
    json_dict = dict()
    with open(json_path, 'r') as jfile:
        json_dict = json.load(jfile)
    return json_dict
