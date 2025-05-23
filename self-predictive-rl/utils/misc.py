#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 31 18:15:44 2025

@author: angel
"""
import sys
import time
import torch
import json
import numpy as np
from typing import Any, Callable, Optional, SupportsFloat, Union, List
from pathlib import Path
import gymnasium as gym
from gymnasium.core import ActType, ObsType

from stable_baselines3.common.monitor import Monitor

from webots_drone.data import StoreStepData
from webots_drone.envs.preprocessor import MultiModalObservation
from webots_drone.envs.preprocessor import CustomVectorObservation
from webots_drone.envs.preprocessor import UAV_DATA
from webots_drone.stack import ObservationStack


class DroneEnvMonitor(Monitor):
    def __init__(self, *args,
                 store_path: Union[Path, str],
                 n_sensors: int = 0,
                 extra_info: bool = True,
                 epsilon_value: Optional[Callable] = None,
                 other_cols: Optional[List[str]] = None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self._data_store = StoreStepData(
            store_path, n_sensors, epsilon_value, extra_info, other_cols)

    def reset(self, **kwargs):# -> tuple[ObsType, dict[str, Any]]:
        obs = super().reset(**kwargs)
        self._data_store.set_init_state(None, obs[1])
        return obs

    def step(self, action: ActType):# -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        observation, reward, terminated, truncated, info = super().step(action)
        sample = (None, action, reward, None, terminated, truncated)
        self._data_store(sample, info)
        return observation, reward, terminated, truncated, info

    def export_env_data(self, outpath=None):
        env_data = {}
        env_data['target_pos'] = str(self.env.unwrapped.vtarget.position)
        env_data['target_quadrants'] = str(self.env.unwrapped.quadrants.tolist())
        env_data['flight_area'] = str(self.env.unwrapped.flight_area.tolist())
        if outpath is None:
            json_path = self._data_store.store_path.parent / 'environment.json'
        else:
            json_path = outpath
        save_dict_json(env_data, json_path)

    def set_episode(self, episode: int = -1):# -> None:
        self._data_store.new_episode(episode)

    def set_eval(self) -> None:
        self._data_store.set_eval()

    def set_learning(self) -> None:
        self._data_store.set_learning()

    def init_store(self) -> None:
        self._data_store.init_store()
        self.export_env_data()


def evaluate_agent(agent_select_action: Callable,
                   env: gym.Env,
                   n_episodes: int,
                   n_steps: int,
                   target_quadrant: int):
    steps = []
    rewards = []
    times = []

    for i in range(n_episodes):
        timemark = time.time()
        state, info = env.reset(target_pos=target_quadrant)
        ep_reward = 0
        ep_steps = 0
        end = False

        while not end:
            action = agent_select_action(state)
            next_state, reward, end, truncated, info = env.step(action)
            # end = done or truncated
            ep_steps += 1
            ep_reward += reward
            state = next_state
            prefix = f"Run {i+1:02d}/{n_episodes:02d}"
            sys.stdout.write(f"\r{prefix} | Reward: {ep_reward:.4f} | "
                             f"Lenght: {ep_steps}  ")
            if ep_steps == n_steps or truncated:
                end = True

        elapsed_time = time.time() - timemark

        steps.append(ep_steps)
        rewards.append(ep_reward)
        times.append(elapsed_time)

    if isinstance(target_quadrant, int):
        target_str = f"{target_quadrant:02d}"
    elif isinstance(target_quadrant, np.ndarray):
        target_str = str(target_quadrant)
    else:
        target_str = 'Random'
    ttime = np.sum(times).round(3)
    tsteps = np.mean(steps)
    treward = np.mean(rewards).round(4)
    sys.stdout.write(f"\r- Evaluated in {ttime:.3f} seconds | "
                     f"Target Position {target_str} | "
                     f"Mean reward: {treward:.4f} | "
                     f"Mean lenght: {tsteps}\n")
    sys.stdout.flush()

    return ep_reward, ep_steps, elapsed_time


def list_of_float(arg):
    return list(map(float, arg.split(',')))


def list_of_int(arg):
    return list(map(int, arg.split(',')))


def list_of_targets(arg):
    if 'random' in arg or 'sample' in arg:
        return arg
    return list_of_int(arg)


def uav_data_list(arg):
    sel_data = list()
    for d in arg.lower().split(','):
        if d in UAV_DATA:
            sel_data.append(d)
    return sel_data


def args2target(env, arg_tpos):
    target_pos = arg_tpos
    if arg_tpos is None:
        target_pos = list(range(len(env.quadrants)))
    elif 'sample' in arg_tpos:
        target_pos = np.random.choice(range(len(env.quadrants)),
                                      int(target_pos.replace('sample-', '')),
                                      replace=False)
    elif arg_tpos == 'random':
        target_pos = [env.vtarget.get_random_position(env.flight_area)]
    elif 'random-' in arg_tpos:
        n_points = int(target_pos.replace('random-', ''))
        target_pos = [env.vtarget.get_random_position(env.flight_area)
                      for _ in range(n_points)]
    return target_pos


def args2env_params(args):
    _args = args
    env_params = {
        'time_limit_seconds': _args.time_limit,
        'max_no_action_seconds': _args.time_no_action,
        'frame_skip': _args.frame_skip,
        'goal_threshold': _args.goal_threshold,
        'init_altitude': _args.init_altitude,
        'altitude_limits': _args.altitude_limits,
        'target_pos': _args.target_pos,
        'target_dim': _args.target_dim,
        'is_pixels': _args.is_pixels,
        'is_vector': _args.is_vector,
        'frame_stack': _args.frame_stack,
        'target_pos2obs': _args.add_target_pos,
        'target_dist2obs': _args.add_target_dist,
        'target_dim2obs': _args.add_target_dim,
        'action2obs': _args.add_action,
        'uav_data': _args.uav_data,
        'norm_obs': True
        }
    env_params['is_multimodal'] = env_params['is_pixels'] and env_params['is_vector']
    return env_params


def instance_env(name='webots_drone:webots_drone/DroneEnvDiscrete-v0',
                 env_params={}, seed=666):
    _env_params = {
        'time_limit_seconds': env_params.get('time_limit', 60),
        'max_no_action_seconds': env_params.get('time_no_action', 5),
        'frame_skip': env_params.get('frame_skip', 6),
        'goal_threshold': env_params.get('goal_threshold', 0.25),
        'init_altitude': env_params.get('init_altitude', 0.3),
        'altitude_limits': env_params.get('altitude_limits', [0.25, 2.]),
        'target_pos': env_params.get('target_pos', None),
        'target_dim': env_params.get('target_dim', [.05, .02]),
        'is_pixels': env_params.get('is_pixels', False),
        }

    if isinstance(_env_params['target_pos'], list):
        if len(_env_params['target_pos']) > 1:
            print('WARNING: Multiple target positions were defined, taking the first one during training.')
        _env_params['target_pos'] = env_params['target_pos'][0]
    # Create the environment
    env = gym.make(name, **_env_params)
    env.seed(seed)

    env_params['state_shape'] = env.observation_space.shape
    if len(env.action_space.shape) == 0:
        env_params['action_shape'] = (env.action_space.n, )
    else:
        env_params['action_shape'] = env.action_space.shape

    # wrap observations
    env_range = {
        'angles_range': [np.pi/2, np.pi/2, np.pi],
        'avel_range': [np.pi, np.pi, 2*np.pi],
        'speed_range': [0.5, 0.5, 0.5]
        }

    if env_params['is_multimodal']:
        env = MultiModalObservation(env, uav_data=env_params['uav_data'],
                                    frame_stack=env_params['frame_stack'],
                                    target_pos=env_params['target_pos2obs'],
                                    target_dim=env_params['target_dim2obs'],
                                    target_dist=env_params['target_dist2obs'],
                                    add_action=env_params['action2obs'],
                                    norm_obs=env_params['norm_obs'],
                                    **env_range)
        env_params['state_shape'] = (env.observation_space['vector'].shape,
                                     env.observation_space['pixel'].shape)

    else:
        if env_params['is_vector']:
            env = CustomVectorObservation(env, uav_data=env_params['uav_data'],
                                          target_dist=env_params['target_dist2obs'],
                                          target_pos=env_params['target_pos2obs'],
                                          target_dim=env_params['target_dim2obs'],
                                          add_action=env_params['action2obs'],
                                          norm_obs=env_params['norm_obs'],
                                          **env_range)

        if env_params['frame_stack'] > 1:
            env = ObservationStack(env, k=env_params['frame_stack'])

        env_params['state_shape'] = env.observation_space.shape

    return env


def save_dict_json(dict2save, json_path):
    proc_dic = dict2save.copy()
    dict_json = json.dumps(proc_dic,
                           indent=4,
                           default=lambda o: str(o))
    with open(json_path, 'w') as jfile:
        jfile.write(dict_json)
    return dict_json
