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
import datetime
import dateutil
from natsort import natsorted
from typing import Any, Callable, Optional, SupportsFloat, Union, List
from pathlib import Path
from collections import namedtuple

import gymnasium as gym
from gymnasium import spaces
from gymnasium.core import ActType, ObsType
from rlpyt.spaces.int_box import IntBox
from stable_baselines3.common.monitor import Monitor

from webots_drone.data import StoreStepData
from webots_drone.envs.preprocessor import CustomVectorObservation
from webots_drone.envs.preprocessor import MultiModalObservation
from webots_drone.envs.preprocessor import UAV_DATA
from webots_drone.stack import ObservationStack


EnvInfo = namedtuple("EnvInfo", ["game_score", "traj_done"])
EnvSpaces = namedtuple("EnvSpaces", ["observation", "action"])


class EnvWrapper(gym.Wrapper[ObsType, ActType, ObsType, ActType]):
    def __init__(self, env: gym.Env, id: int = 0):
        super().__init__(env=env)
        control_limits = np.hstack((self.env.action_limits[1],
                                    self.env.action_limits[0][::-1]))
        self.action_space = IntBox(low=0, high=control_limits.shape[-1] + 1)
        # Observation space, the drone's camera image
        # self.obs_shape = (1, 9, 84, 84)
        # self.obs_type = np.uint8
        # self.observation_space = spaces.Box(low=0,
        #                                     high=255,
        #                                     shape=self.obs_shape,
        #                                     dtype=self.obs_type)

    @property
    def spaces(self):
        return EnvSpaces(
            observation=self.observation_space,
            action=self.action_space,
        )

    def get_observation_2d(self, state_data, norm=False):
        state_2d = self.env.get_observation_2d(state_data, norm)
        return state_2d[np.newaxis, :]

    def reset(self, seed=None, target_pos=None, target_dim=None, **kwargs):
        obs, _ = self.env.reset(seed=seed, target_pos=target_pos, target_dim=target_dim, **kwargs)
        return obs

    def step(self, action):
        observation, reward, end, truncated, _ = self.env.step(action)
        info = EnvInfo(game_score=reward, traj_done=truncated or end)
        return observation, reward, end or truncated, info


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

    def reset(self, **kwargs) -> tuple[ObsType, dict[str, Any]]:
        obs = super().reset(**kwargs)
        self._data_store.set_init_state(None, obs[1])
        return obs

    def step(self, action: ActType) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        observation, reward, terminated, truncated, info = super().step(action)
        sample = (None, action.item(), reward, None, terminated, truncated)
        self._data_store(sample, info)
        return observation, reward, terminated, truncated, info

    def export_env_data(self, outpath=None):
        env_data = {}
        env_data['target_quadrants'] = str(self.env.quadrants.tolist())
        env_data['flight_area'] = str(self.env.flight_area.tolist())
        if outpath is None:
            json_path = self._data_store.store_path / 'environment.json'
        else:
            json_path = outpath
        save_dict_json(env_data, json_path)

    def set_episode(self, episode: int = -1) -> None:
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
        state = env.reset(target_pos=target_quadrant)
        ep_reward = 0
        ep_steps = 0
        end = False

        while not end:
            action = agent_select_action(state)
            next_state, reward, end, info = env.step(action)
            # end = done or truncated
            ep_steps += 1
            ep_reward += reward
            state = next_state
            prefix = f"Run {i+1:02d}/{n_episodes:02d}"
            sys.stdout.write(f"\r{prefix} | Reward: {ep_reward:.4f} | "
                             f"Length: {ep_steps}  ")
            if ep_steps == n_steps or info.traj_done:
                end = True

        elapsed_time = time.time() - timemark

        steps.append(ep_steps)
        rewards.append(ep_reward)
        times.append(elapsed_time)

    if isinstance(target_quadrant, int):
        target_str = f"{target_quadrant:02d}"
    elif isinstance(target_quadrant, np.ndarray):
        target_str = str(target_quadrant)
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


def parse_crazyflie_env_args(parser):
    arg_env = parser.add_argument_group('Environment')
    arg_env.add_argument("--time-limit", type=int, default=60,  # 1m
                         help='Max time (seconds) of the mission.')
    arg_env.add_argument("--time-no-action", type=int, default=5,
                         help='Max time (seconds) with no movement.')
    arg_env.add_argument("--frame-skip", type=int, default=6,  # 192ms
                         help='Number of simulation steps for a RL step')
    arg_env.add_argument("--frame-stack", type=int, default=1,
                         help='Number of RL step to stack as observation.')
    arg_env.add_argument("--goal-threshold", type=float, default=0.25,
                         help='Minimum distance from the target.')
    arg_env.add_argument("--init-altitude", type=float, default=1.0,
                         help='Minimum height distance to begin the mission.')
    arg_env.add_argument("--altitude-limits", type=list_of_float,
                         default=[0.25, 2.], help='Vertical flight limits.')
    arg_env.add_argument("--target-pos", type=list_of_targets, default=None,
                         help='Cuadrant number for target position.')
    arg_env.add_argument("--target-dim", type=list_of_float, default=[0.05, 0.02],
                         help="Target's dimension size.")
    arg_env.add_argument("--zone-steps", type=int, default=0,
                         help='Max number on target area to end the episode with found target.')
    arg_env.add_argument("--is-pixels", action='store_true',
                         help='Whether if reconstruct an image-based observation.')
    arg_env.add_argument("--is-vector", action='store_true',
                         help='Whether if reconstruct a vector-based observation.')
    arg_env.add_argument("--add-target-pos", action='store_true',
                         help='Whether if add the target position to vector state.')
    arg_env.add_argument("--add-target-dist", action='store_true',
                         help='Whether if add the target distance to vector state.')
    arg_env.add_argument("--add-target-dim", action='store_true',
                         help='Whether if add the target dimension to vector state.')
    arg_env.add_argument("--add-action", action='store_true',
                         help='Whether if add the previous action to vector state.')
    arg_env.add_argument("--uav-data", type=uav_data_list,
                         default=['imu', 'gyro', 'gps', 'gps_vel', 'north', 'dist_sensors'],
                         help='Select the UAV sensor data as state, available'
                         ' options are: imu, gyro, gps, gps_vel, north, dist_sensors')
    return arg_env


def parse_alg_args(parser):
    parser.add_argument('--game', help='Atari game', default='ms_pacman')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--grayscale', type=int, default=1)
    parser.add_argument('--framestack', type=int, default=3)
    parser.add_argument('--imagesize', type=int, default=84)
    parser.add_argument('--n-steps', type=int, default=450000)
    parser.add_argument('--dqn-hidden-size', type=int, default=32)
    parser.add_argument('--target-update-interval', type=int, default=1)
    parser.add_argument('--target-update-tau', type=float, default=1.)
    parser.add_argument('--momentum-tau', type=float, default=0.01)
    parser.add_argument('--batch-b', type=int, default=1)
    parser.add_argument('--batch-t', type=int, default=1)
    parser.add_argument('--beluga', action="store_true")
    parser.add_argument('--jumps', type=int, default=5)
    parser.add_argument('--num-logs', type=int, default=50)
    parser.add_argument('--renormalize', type=int, default=1)
    parser.add_argument('--dueling', type=int, default=1)
    parser.add_argument('--replay-ratio', type=int, default=64)
    parser.add_argument('--dynamics-blocks', type=int, default=0)
    parser.add_argument('--residual-tm', type=int, default=0.)
    parser.add_argument('--n-step', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--tag', type=str, default='', help='Tag for wandb run.')
    parser.add_argument('--wandb-dir', type=str, default='', help='Directory for wandb files.')
    parser.add_argument('--norm-type', type=str, default='bn', choices=["bn", "ln", "in", "none"], help='Normalization')
    parser.add_argument('--aug-prob', type=float, default=1., help='Probability to apply augmentation')
    parser.add_argument('--dropout', type=float, default=0., help='Dropout probability in convnet.')
    parser.add_argument('--spr', type=int, default=1)
    parser.add_argument('--distributional', type=int, default=1)
    parser.add_argument('--delta-clip', type=float, default=1., help="Huber Delta")
    parser.add_argument('--prioritized-replay', type=int, default=1)
    parser.add_argument('--momentum-encoder', type=int, default=1)
    parser.add_argument('--shared-encoder', type=int, default=0)
    parser.add_argument('--local-spr', type=int, default=0)
    parser.add_argument('--global-spr', type=int, default=1)
    parser.add_argument('--noisy-nets', type=int, default=1)
    parser.add_argument('--noisy-nets-std', type=float, default=0.5)
    parser.add_argument('--classifier', type=str, default='q_l1', choices=["mlp", "bilinear", "q_l1", "q_l2", "none"], help='Style of NCE classifier')
    parser.add_argument('--final-classifier', type=str, default='linear', choices=["mlp", "linear", "none"], help='Style of NCE classifier')
    parser.add_argument('--augmentation', type=str, default=["shift", "intensity"], nargs="+",
                        choices=["none", "rrc", "affine", "crop", "blur", "shift", "intensity"],
                        help='Style of augmentation')
    parser.add_argument('--q-l1-type', type=str, default=["value", "advantage"], nargs="+",
                        choices=["noisy", "value", "advantage", "relu"],
                        help='Style of q_l1 projection')
    parser.add_argument('--target-augmentation', type=int, default=1, help='Use augmentation on inputs to target networks')
    parser.add_argument('--eval-augmentation', type=int, default=0, help='Use augmentation on inputs at evaluation time')
    parser.add_argument('--reward-loss-weight', type=float, default=0.)
    parser.add_argument('--model-rl-weight', type=float, default=0.)
    parser.add_argument('--model-spr-weight', type=float, default=5.)
    parser.add_argument('--t0-spr-loss-weight', type=float, default=0.)
    parser.add_argument('--eps-steps', type=int, default=2001)
    parser.add_argument('--min-steps-learn', type=int, default=2000)
    parser.add_argument('--eps-init', type=float, default=1.)
    parser.add_argument('--eps-final', type=float, default=0.)
    parser.add_argument('--final-eval-only', type=int, default=1)
    parser.add_argument('--time-offset', type=int, default=0)
    parser.add_argument('--project', type=str, default="mpr")
    parser.add_argument('--entity', type=str, default="abs-world-models")
    parser.add_argument('--cuda_idx', help='gpu to use ', type=int, default=0)
    parser.add_argument('--max-grad-norm', type=float, default=10., help='Max Grad Norm')
    parser.add_argument('--public', action='store_true', help='If set, uses anonymous wandb logging')


def args2env_params(args):
    _args = args
    if not isinstance(_args, dict):
        _args = vars(_args)
    env_params = {
        'time_limit_seconds': _args.get('time_limit', 60),
        'max_no_action_seconds': _args.get('time_no_action', 5),
        'frame_skip': _args.get('frame_skip', 6),
        'goal_threshold': _args.get('goal_threshold', 0.25),
        'init_altitude': _args.get('init_altitude', 0.3),
        'altitude_limits': _args.get('altitude_limits', [0.25, 2.]),
        'target_pos': _args.get('target_pos', None),
        'target_dim': _args.get('target_dim', [.05, .02]),
        'is_pixels': _args.get('is_pixels', False),
        'is_vector': _args.get('is_vector', False),
        'frame_stack': _args.get('frame_stack', 1),
        'target_pos2obs': _args.get('add_target_pos', False),
        'target_dist2obs': _args.get('add_target_dist', False),
        'target_dim2obs': _args.get('add_target_dim', False),
        'action2obs': _args.get('add_action', False),
        'uav_data': _args.get('uav_data',
                              ['imu', 'gyro', 'gps', 'gps_vel', 'north', 'dist_sensors']),
        }
    env_params['is_multimodal'] = env_params['is_pixels'] and env_params['is_vector']
    return env_params


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


def instance_env(name='webots_drone:webots_drone/DroneEnvDiscrete-v0',
                 env_args={}, seed=666):
    env_params = {
        'time_limit_seconds': env_args.get('time_limit', 60),
        'max_no_action_seconds': env_args.get('time_no_action', 5),
        'frame_skip': env_args.get('frame_skip', 6),
        'goal_threshold': env_args.get('goal_threshold', 0.25),
        'init_altitude': env_args.get('init_altitude', 0.3),
        'altitude_limits': env_args.get('altitude_limits', [0.25, 2.]),
        'target_pos': env_args.get('target_pos', None),
        'target_dim': env_args.get('target_dim', [.05, .02]),
        'is_pixels': env_args.get('is_pixels', False),
        }

    if isinstance(env_params['target_pos'], list):
        if len(env_params['target_pos']) > 1:
            print('WARNING: Multiple target positions were defined, taking the first one during training.')
        env_params['target_pos'] = env_params['target_pos'][0]
    # Create the environment
    env = gym.make(name, **env_params)
    env.seed(seed)

    env_args['state_shape'] = env.observation_space.shape
    if len(env.action_space.shape) == 0:
        env_args['action_shape'] = (env.action_space.n, )
    else:
        env_args['action_shape'] = env.action_space.shape

    return env


def wrap_env(env, env_params):
    env_range = {
        'angles_range': [np.pi/6, np.pi/6, np.pi],
        'avel_range': [np.pi/3, np.pi/3, 2*np.pi],
        'speed_range': [0.8, 0.8, 0.6]
        }
    if env_params['is_multimodal']:
        env = MultiModalObservation(env, uav_data=env_params['uav_data'],
                                    frame_stack=env_params['frame_stack'],
                                    target_pos=env_params['target_pos2obs'],
                                    target_dim=env_params['target_dim2obs'],
                                    target_dist=env_params['target_dist2obs'],
                                    add_action=env_params['action2obs'],
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
                                          **env_range)

        if env_params['frame_stack'] > 1:
            env = ObservationStack(env, k=env_params['frame_stack'], add_t=True, force_np=True)

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


def load_json_dict(json_path):
    json_dict = dict()
    with open(json_path, 'r') as jfile:
        json_dict = json.load(jfile)
    return json_dict


def load_agent(args, params_path='logs/run_0/params.pkl'):
    from src.algos import SPRCategoricalDQN
    from src.agent import SPRAgent
    from src.models import SPRCatDqnModel
    from src.utils import set_config

    data = torch.load(params_path)
    agent_state_dict = data['agent_state_dict']  # 'model' and 'target' keys
    optimizer_state_dict = data['optimizer_state_dict']

    config = set_config(args, args.game)
    args.discount = config["algo"]["discount"]
    algo = SPRCategoricalDQN(
        optim_kwargs=config["optim"], jumps=args.jumps,
        initial_optim_state_dict=optimizer_state_dict,
        **config["algo"])  # Run with defaults.
    agent = SPRAgent(ModelCls=SPRCatDqnModel, model_kwargs=config["model"],
                     initial_model_state_dict=agent_state_dict['model'],
                     **config["agent"])
    agent.set_eval_on = lambda : agent.eval_mode(data['itr'])
    return agent


def iterate_agents_evaluation(env, args):
    logs_path = Path(args.logspath)
    agent_models = natsorted(logs_path.glob('*.pkl'), key=str)

    for log_ep, agent_path in enumerate(agent_models):
        if args.episode > -1 and log_ep != args.episode:
            continue
        # custom agent episodes selection
        elif args.episode == -1 and log_ep not in [5, 10, 20, 35, 50]:
            continue

        print('Loading', agent_path)
        agent = load_agent(args, agent_path)

        def action_selection(observations):
            agent_action = agent.step(torch.from_numpy(observations), None, None)
            return agent_action.action

        # Target position for evaluation
        targets_pos = args2target(env, args.target_pos)
        # Log eval data
        ep_name = agent_path.stem.replace('itr_', '')
        csv_path = logs_path / 'eval' / f"history_{ep_name}.csv"
        csv_path.parent.mkdir(exist_ok=True)
        monitor_env = DroneEnvMonitor(env, store_path=csv_path, n_sensors=4,
                                      reset_keywords=['target_pos'])
        monitor_env.init_store()
        monitor_env.set_eval()
        monitor_env = EnvWrapper(monitor_env)  # rlpy compat env

        agent.initialize(monitor_env.spaces)
        agent.set_eval_on()
        # Iterate over goal position
        monitor_env.set_episode(log_ep)
        for tpos in targets_pos:
            evaluate_agent(action_selection, monitor_env, args.eval_episodes,
                           args.eval_steps, tpos)
        monitor_env.close()

def now_str():
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    return now.strftime("%Y-%m-%d_%H-%M-%S")
