#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 19 16:08:22 2025

@author: angel
"""

from typing import Any, Callable, Optional, List, SupportsFloat, Union
from gymnasium.core import ActType, ObsType
import sys
import time
import numpy as np
import gymnasium as gym
from pathlib import Path
from natsort import natsorted

from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from webots_drone.data import StoreStepData
from webots_drone.envs.preprocessor import MultiModalObservation
from webots_drone.envs.preprocessor import CustomVectorObservation
from webots_drone.envs.preprocessor import UAV_DATA
from webots_drone.stack import ObservationStack

from .agent import save_dict_json


def list_of_float(arg):
    return list(map(float, arg.split(',')))


def list_of_int(arg):
    return list(map(int, arg.split(',')))


def target_args(arg):
    if 'random' in arg or 'sample' in arg:
        return arg
    return list_of_int(arg)


def uav_data_list(arg):
    global UAV_DATA
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
    arg_env.add_argument("--target-dim", type=list_of_float, default=[0.05, 0.02],
                         help="Target's dimension size.")
    parse_common_env_args(arg_env)
    return arg_env


def parse_mavic_env_args(parser):
    arg_env = parser.add_argument_group('Environment')
    arg_env.add_argument("--time-limit", type=int, default=600,  # 10m
                         help='Max time (seconds) of the mission.')
    arg_env.add_argument("--time-no-action", type=int, default=5,
                         help='Max time (seconds) with no movement.')
    arg_env.add_argument("--frame-skip", type=int, default=25,  # 200ms
                         help='Number of simulation steps for a RL step')
    arg_env.add_argument("--frame-stack", type=int, default=1,
                         help='Number of RL step to stack as observation.')
    arg_env.add_argument("--goal-threshold", type=float, default=5.,
                         help='Minimum distance from the target.')
    arg_env.add_argument("--init-altitude", type=float, default=25.,
                         help='Minimum height distance to begin the mission.')
    arg_env.add_argument("--altitude-limits", type=list_of_float,
                         default=[11., 75.], help='Vertical flight limits.')
    arg_env.add_argument("--target-dim", type=list_of_float, default=[7., 3.5],
                         help="Target's dimension size.")
    parse_common_env_args(arg_env)
    return arg_env


def parse_common_env_args(arg_env):
    arg_env.add_argument("--target-pos", type=target_args, default=None,
                         help='Cuadrant number for target position.')
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
    arg_env.add_argument("--uav-data", type=uav_data_list, default=UAV_DATA,
                         help='Select the UAV sensor data as state, available'
                         ' options are: imu, gyro, gps, gps_vel, north, dist_sensors')
    arg_env.add_argument("--normalized-obs", action='store_true',
                         help='Whether if use a normalized observation [-1, 1].')
    return arg_env


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
        'uav_data': _args.get('uav_data', UAV_DATA),
        'norm_obs': _args.get('normalized_obs', False),
        }
    env_params['is_multimodal'] = env_params['is_pixels'] and env_params['is_vector']
    return env_params


def instance_env(name='webots_drone:webots_drone/DroneEnvDiscrete-v0',
                 env_params={}, seed=666, extra_params=None):
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
    if extra_params:
        _env_params.update(extra_params)

    if isinstance(_env_params['target_pos'], list):
        if len(_env_params['target_pos']) > 1:
            print('WARNING: Multiple target positions were defined, taking the first one during training.')
        _env_params['target_pos'] = env_params['target_pos'][0]
    # Create the environment
    env = gym.make(name, **_env_params)
    env.unwrapped.seed(seed)

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


def args2target(env, arg_tpos):
    target_pos = arg_tpos
    if arg_tpos is None:
        target_pos = list(range(len(env.unwrapped.quadrants)))
    elif isinstance(arg_tpos, int):
        target_pos = [arg_tpos]
    elif 'sample' in arg_tpos:
        target_pos = np.random.choice(range(len(env.unwrapped.quadrants)),
                                      int(target_pos.replace('sample-', '')),
                                      replace=False)
    elif arg_tpos == 'random':
        target_pos = [env.vtarget.get_random_position(env.flight_area)]
    elif 'random-' in arg_tpos:
        n_points = int(target_pos.replace('random-', ''))
        target_pos = [env.vtarget.get_random_position(env.flight_area)
                      for _ in range(n_points)]
    return target_pos


class DroneExperimentCallback(CheckpointCallback):
    """
    A callback to save the arguments after creating the log output folder.

    :param exp_args: The Parser object to be save.
    :param out_path: The json file path to be writen.
    :param memory_steps: The number of steps to initialize the memory.
    :param data_store: The StepDataStore object.
    """

    def __init__(self, *args,
                 env: gym.Env,
                 exp_args: dict,
                 out_path: str,
                 memory_steps: int,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.exp_args = vars(exp_args)
        self.out_path = out_path
        self.env = env
        # apply a offset to ensure saving agents after save_freq without memory fill
        self.n_calls = -memory_steps
        self.save_freq_tmp = self.save_freq
        self.save_freq = float('inf')

    def _on_training_start(self) -> None:
        if isinstance(self.env.action_space, gym.spaces.Box):
            self.exp_args['action_shape'] = self.env.action_space.shape
            self.exp_args['action_high'] = self.env.action_space.high
            self.exp_args['action_low'] = self.env.action_space.low
        if isinstance(self.env.action_space, gym.spaces.Discrete):
            self.exp_args['action_shape'] = self.env.action_space.n
        self.exp_args['obs_shape'] = self.env.observation_space.shape
        save_dict_json(self.exp_args, self.out_path)
        self.env.init_store()

    def _on_step(self) -> bool:
        if self.n_calls == 0:
            self.env.set_learning()
            self.save_freq = self.save_freq_tmp
        if self.n_calls % self.save_freq == 0:
            self.env.new_episode()
            self.training_env.reset()
        return super()._on_step()


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
        sample = (None, action, reward, None, terminated, truncated)
        self._data_store(sample, info)
        return observation, reward, terminated, truncated, info

    def export_env_data(self, outpath: Optional[Union[str, Path]] = None) -> None:
        env_data = {}
        env_data['target_quadrants'] = str(self.env.unwrapped.quadrants.tolist())
        env_data['flight_area'] = str(self.env.unwrapped.flight_area.tolist())
        env_data['action_limits'] = str(self.env.unwrapped.action_limits.tolist())
        if outpath is None:
            json_path = self._data_store.store_path.parent / 'environment.json'
        else:
            json_path = outpath
        save_dict_json(env_data, json_path)

    def new_episode(self, episode: int = -1) -> None:
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
            next_state, reward, done, truncated, info = env.step(action)
            end = done or truncated
            ep_steps += 1
            ep_reward += reward
            state = next_state
            prefix = f"Run {i+1:02d}/{n_episodes:02d}"
            sys.stdout.write(f"\r{prefix} | Reward: {ep_reward:.4f} | "
                             f"Length: {ep_steps}  ")
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


def iterate_agents_evaluation(env, algorithm, args, log_args=None):
    logs_path = Path(args.logspath)
    agent_models = natsorted(logs_path.glob('agents/rl_model_*'), key=str)

    for log_ep, agent_path in enumerate(agent_models):
        if args.episode > -1 and log_ep != args.episode:
            continue
        # custom agent episodes selection
        elif args.episode == -1 and log_ep not in [5, 10, 20, 35, 50]:
            continue

        print('Loading', agent_path)
        model = algorithm.load(agent_path)
        def action_selection(observations):
            if type(observations) is dict:
                for k in observations.keys():
                    observations[k] = np.array(observations[k], dtype=np.float32)
                    if observations[k].shape[0] != 1:
                        observations[k] = observations[k][np.newaxis, ...]
            else:
                observations = np.array(observations, dtype=np.float32)
                if observations.shape[0] != 1:
                    observations = observations[np.newaxis, ...]
            actions, states = model.predict(
                observations,  # type: ignore[arg-type]
                state=None,
                episode_start=None,
                deterministic=True,
            )
            return actions[0]

        # Target position for evaluation
        targets_pos = args2target(env, args.target_pos)
        # Log eval data
        if log_args is None or 'store_path' not in log_args.keys():
            out_path = logs_path / 'eval'
        else:
            out_path = Path(log_args['store_path'])
            if not out_path.is_dir():
                out_path = out_path.parent
        out_path.mkdir(exist_ok=True)
        if log_args is None:
            log_args = {'n_sensors': 4,
                        'reset_keywords': ['target_pos']}
        else:
            log_args['reset_keywords'] = ['target_pos']

        ep_name = agent_path.stem.replace('rl_model_', '')
        log_args['store_path'] = out_path / f"history_{ep_name}.csv"
        # Instantiate Monitor
        monitor_env = DroneEnvMonitor(env, **log_args)
        monitor_env.init_store()
        monitor_env.set_eval()
        monitor_env.new_episode(log_ep)
        # Iterate over goal position
        for tpos in targets_pos:
            evaluate_agent(action_selection, monitor_env, args.eval_episodes,
                           args.eval_steps, tpos)
        monitor_env.close()
