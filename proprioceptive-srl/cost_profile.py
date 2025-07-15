#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 18:31:16 2024

@author: Angel Ayala
"""
import argparse
import torch
from pathlib import Path
from thop import clever_format, profile
from natsort import natsorted

from stable_baselines3 import TD3
from stable_baselines3.td3.policies import TD3Policy
from stable_baselines3.td3.policies import CnnPolicy as TD3CnnPolicy
from stable_baselines3.td3.policies import MultiInputPolicy as TD3MultiInputPolicy
from sb3_srl.td3_srl import SRLTD3Policy, SRLTD3

from stable_baselines3 import SAC
from stable_baselines3.sac.policies import SACPolicy
from stable_baselines3.sac.policies import CnnPolicy as SACCnnPolicy
from stable_baselines3.sac.policies import MultiInputPolicy as SACMultiInputPolicy
from sb3_srl.sac_srl import SRLSACPolicy, SRLSAC

from utils import (
    args2ae_config,
    args2env_params,
    instance_env,
    load_json_dict
    )


def parse_args():
    parser = argparse.ArgumentParser()    # misc
    parser.add_argument('--logspath', type=str,
                        default='logs/ddqn-srl_2023-11-28_00-13-33',
                        help='Log path with training results.')
    parser.add_argument('--episode', type=int, default=-1,
                          help='Indicate the episode number to execute, set -1 for all of them')
    parser.add_argument("--with-model", action='store_true',
                         help="Whether if print the model's architecture.")
    args = parser.parse_args()
    return args


def profile_model(model, input_shape, device, action_shape=None):
    """Profiling developed models.

    based on https://github.com/angel-ayala/kutralnet/blob/master/utils/profiling.py"""
    x = torch.randn(input_shape).unsqueeze(0).to(device)
    if action_shape:
        y = torch.randn(action_shape).unsqueeze(0).to(device)
        flops, params = profile(model, verbose=False,
                                inputs=(x, y),)
    else:
        flops, params = profile(model, verbose=False,
                                inputs=(x, ),)
    return flops, params


if __name__ == '__main__':
    # Load arguments
    eval_args = parse_args()
    is_sac = 'sac' in str(eval_args.logspath).lower()
    saved_args = load_json_dict(eval_args.logspath + '/arguments.json')
    env_params = args2env_params(saved_args)

    # Algorithm
    if saved_args['is_srl']:
        if is_sac:
            algorithm, policy = SRLSAC, SRLSACPolicy
        else:
            algorithm, policy = SRLTD3, SRLTD3Policy
        # Autoencoder parameters
        env_params['action_shape'] = (4, )  # TODO: read from json file
        env_params['state_shape'] = (18, )
        ae_config = args2ae_config(saved_args, env_params)

        # Policy args
        policy_args = {
            'ae_config': ae_config,
            'encoder_tau': saved_args['encoder_tau']
            }
        if not is_sac:
            policy_args['net_arch'] = [256, 256]
    else:
        if is_sac:
            algorithm, policy = SAC, SACPolicy
        else:
            algorithm, policy = TD3, TD3Policy
        policy_args = None
        if saved_args['is_pixels']:
            policy = SACCnnPolicy if is_sac else TD3CnnPolicy
            if saved_args['is_vector']:
                policy = SACMultiInputPolicy if is_sac else TD3MultiInputPolicy

    # Instantiate algorithm
    logs_path = Path(eval_args.logspath)
    agent_models = natsorted(logs_path.glob('agents/rl_model_*'), key=str)
    agent_path = agent_models[eval_args.episode]
    print('Loading from', "/".join(str(agent_path).split("/")[-3:]))
    model = algorithm.load(agent_path)
    is_srl = hasattr(model.policy, 'rep_model')

    # Architecture
    print("====== Policy's architecture ======")
    policy = model.policy
    learn_flops, learn_params = 0, 0
    eval_flops, eval_params = 0, 0
    print("PolicyClass:", policy.__class__.__name__)
    print("* CriticClass:", policy.critic.__class__.__name__)
    print("  - StateSpace:", policy.critic.observation_space.shape)
    critic_input = (policy.actor.features_dim,
                    ) if is_srl else policy.critic.observation_space.shape
    flops, params = profile_model(
        policy.critic, critic_input, model.device,
        action_shape=model.policy.actor.action_space.shape)
    n_critics = len(policy.critic.q_networks)
    learn_flops += flops * 2
    learn_params += params * 2
    eval_flops += flops / n_critics
    eval_params += params / n_critics
    flops_str, params_str = clever_format([flops, params], "%.3f")
    print("  - Learning cost:",
          f" {n_critics} X {flops_str} flops, {params_str} params ({model.device})")
    flops_str, params_str = clever_format(
        [flops / n_critics, params / n_critics], "%.3f")
    print("  - Eval cost:    ",
          f" {flops_str} flops, {params_str} params ({model.device})")
    if eval_args.with_model:
        print("  - Model:", policy.critic)

    print("* ActorClass:", policy.actor.__class__.__name__)
    print("  - ActionSpace:", policy.actor.action_space.shape)
    flops, params = profile_model(
        policy.actor, policy.actor.features_dim, model.device)
    n_actors = 1 if is_sac else 2
    learn_flops += flops * n_actors
    learn_params += params * n_actors
    eval_flops += flops
    eval_params += params
    flops_str, params_str = clever_format([flops, params], "%.3f")
    print("  - Learning cost:",
          f" {n_actors} X {flops_str} flops, {params_str} params ({model.device})")
    print("  - Eval cost:    ",
          f" {flops_str} flops, {params_str} params ({model.device})")
    if eval_args.with_model:
        print("  - Model:", policy.actor)

    if is_srl:
        print()
        print("====== Representation's architecture ======")
        print(model.policy.rep_model)
        encoder = policy.rep_model.encoder
        decoder = policy.rep_model.decoder
        print("* EncoderClass:", encoder.__class__.__name__)
        print("  - LatentDim:", encoder.latent_dim)
        encoder_input = policy.critic.observation_space.shape
        flops, params = profile_model(encoder, encoder_input, model.device)
        learn_flops += flops * 2
        learn_params += params * 2
        eval_flops += flops
        eval_params += params
        flops_str, params_str = clever_format([flops * 2, params * 2], "%.3f")
        print("  - Learning cost:",
              f" 2 X {flops_str} flops, {params_str} params ({model.device})")
        flops_str, params_str = clever_format([flops, params], "%.3f")
        print("  - Eval cost:    ",
              f" {flops_str} flops, {params_str} params ({model.device})")
        if eval_args.with_model:
            print("  - Model:", encoder)

        print("* DecoderClass:", decoder.__class__.__name__)
        print("  - OutputDim:", policy.actor.features_dim)
        decoder_input = policy.actor.features_dim
        flops, params = profile_model(
            decoder, decoder_input, model.device,
            action_shape=model.policy.actor.action_space.shape)
        learn_flops += flops
        learn_params += params
        flops_str, params_str = clever_format([flops, params], "%.3f")
        print("  - Learning cost:",
              f" {flops_str} flops, {params_str} params ({model.device})")
        flops_str, params_str = clever_format([0, 0], "%.3f")
        print("  - Eval cost:    ",
              f" {flops_str} flops, {params_str} params ({model.device})")
        if eval_args.with_model:
            print("  - Model:", decoder)

    print()
    print("="*60)
    flops_str, params_str = clever_format([learn_flops, learn_params], "%.3f")
    print("Total cost learning:  ",
          f" {flops_str} flops, {params_str} params ({model.device})")
    flops_str, params_str = clever_format([eval_flops, eval_params], "%.3f")
    print("Total cost evaluating:",
          f" {flops_str} flops, {params_str} params ({model.device})")
    print()
