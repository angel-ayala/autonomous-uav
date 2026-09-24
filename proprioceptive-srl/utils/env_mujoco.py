#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 19 16:32:40 2025

@author: angel
"""
from typing import Any, Callable
from datetime import datetime
import time
import json
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.core import (
    ActType,
    ObsType,
    WrapperObsType
    )
from pathlib import Path
from natsort import natsorted
import sys

from sb3_srl.agent_utils import parse_training_args
from sb3_srl.agent_utils import save_dict_json

from stable_baselines3.common.callbacks import EvalCallback

from .agent import evaluate_agent


def mujoco_training_args(parser):
    arg_training = parse_training_args(
        parser,
        steps=500000,
        memory_steps=5000,
        batch_size=512,
        eval_interval=10000,
        eval_steps=1000)
    arg_training.add_argument('--eval-episodes', type=int, default=10,  # 1m at 25 frames
                              help='Number of evaluation steps.')
    return arg_training


def parse_mujoco_env_args(parser):
    arg_env = parser.add_argument_group('Environment')
    arg_env.add_argument("--environment-id", type=str, default="Ant-v2",
                         help='The Mujoco control environment name ID.')
    return arg_env


def get_env(env_id: str, seed: int = 666, render_mode=None):
    env = gym.make(env_id, render_mode=render_mode)

    if env.observation_space.dtype == np.float64:
        env = DtypeObservation(env, np.float32)

    env = gym.wrappers.RecordEpisodeStatistics(env)
    # env.seed(seed=seed)
    env.observation_space.seed(seed)
    env.action_space.seed(seed)
    return env


class TransformObservation(
    gym.ObservationWrapper[WrapperObsType, ActType, ObsType],
    gym.utils.RecordConstructorArgs,
):
    """Applies a function to the ``observation`` received from the environment's :meth:`Env.reset` and :meth:`Env.step` that is passed back to the user.

    The function :attr:`func` will be applied to all observations.
    If the observations from :attr:`func` are outside the bounds of the ``env``'s observation space, provide an updated :attr:`observation_space`.

    A vector version of the wrapper exists :class:`gymnasium.wrappers.vector.TransformObservation`.

    Example:
        >>> import gymnasium as gym
        >>> from gymnasium.wrappers import TransformObservation
        >>> import numpy as np
        >>> np.random.seed(0)
        >>> env = gym.make("CartPole-v1")
        >>> env.reset(seed=42)
        (array([ 0.0273956 , -0.00611216,  0.03585979,  0.0197368 ], dtype=float32), {})
        >>> env = gym.make("CartPole-v1")
        >>> env = TransformObservation(env, lambda obs: obs + 0.1 * np.random.random(obs.shape), env.observation_space)
        >>> env.reset(seed=42)
        (array([0.08227695, 0.06540678, 0.09613613, 0.07422512]), {})

    Change logs:
     * v0.15.4 - Initially added
     * v1.0.0 - Add requirement of ``observation_space``
    """

    def __init__(
        self,
        env: gym.Env[ObsType, ActType],
        func: Callable[[ObsType], Any],
        observation_space: gym.Space[WrapperObsType] | None,
    ):
        """Constructor for the transform observation wrapper.

        Args:
            env: The environment to wrap
            func: A function that will transform an observation. If this transformed observation is outside the observation space of ``env.observation_space`` then provide an `observation_space`.
            observation_space: The observation spaces of the wrapper, if None, then it is assumed the same as ``env.observation_space``.
        """
        gym.utils.RecordConstructorArgs.__init__(
            self, func=func, observation_space=observation_space
        )
        gym.ObservationWrapper.__init__(self, env)

        if observation_space is not None:
            self.observation_space = observation_space

        self.func = func

    def observation(self, observation: ObsType) -> Any:
        """Apply function to the observation."""
        return self.func(observation)


class DtypeObservation(
    TransformObservation[WrapperObsType, ActType, ObsType],
    gym.utils.RecordConstructorArgs,
):
    """Modifies the dtype of an observation array to a specified dtype.

    Note:
        This is only compatible with :class:`Box`, :class:`Discrete`, :class:`MultiDiscrete` and :class:`MultiBinary` observation spaces

    A vector version of the wrapper exists :class:`gymnasium.wrappers.vector.DtypeObservation`.

    Change logs:
     * v1.0.0 - Initially added
    """

    def __init__(self, env: gym.Env[ObsType, ActType], dtype: Any):
        """Constructor for Dtype observation wrapper.

        Args:
            env: The environment to wrap
            dtype: The new dtype of the observation
        """
        assert isinstance(
            env.observation_space,
            (spaces.Box, spaces.Discrete, spaces.MultiDiscrete, spaces.MultiBinary),
        )

        self.dtype = dtype
        if isinstance(env.observation_space, spaces.Box):
            new_observation_space = spaces.Box(
                low=env.observation_space.low,
                high=env.observation_space.high,
                shape=env.observation_space.shape,
                dtype=self.dtype,
            )
        elif isinstance(env.observation_space, spaces.Discrete):
            new_observation_space = spaces.Box(
                low=env.observation_space.start,
                high=env.observation_space.start + env.observation_space.n,
                shape=(),
                dtype=self.dtype,
            )
        elif isinstance(env.observation_space, spaces.MultiDiscrete):
            new_observation_space = spaces.MultiDiscrete(
                env.observation_space.nvec, dtype=dtype
            )
        elif isinstance(env.observation_space, spaces.MultiBinary):
            new_observation_space = spaces.Box(
                low=0,
                high=1,
                shape=env.observation_space.shape,
                dtype=self.dtype,
            )
        else:
            raise TypeError(
                "DtypeObservation is only compatible with value / array-based observations."
            )

        gym.utils.RecordConstructorArgs.__init__(self, dtype=dtype)
        TransformObservation.__init__(
            self,
            env=env,
            func=lambda obs: dtype(obs),
            observation_space=new_observation_space,
        )


class CustomEvalCallback(EvalCallback):
    def __init__(self, *args,
                 args_exp: dict = None,
                 args_path: str = None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.args_exp = vars(args_exp)
        self.args_path = args_path

    def _init_callback(self) -> None:
        super()._init_callback()
        save_dict_json(self.args_exp, self.args_path)


class EvaluationRecorder(gym.Wrapper):

    def __init__(self, env, path):
        super().__init__(env)

        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)

        self.evaluations = []
        self.video_env = None

    def start_video(self, name_prefix):
        self.video_end()

        self.video_env = gym.wrappers.RecordVideo(
            self.env,
            video_folder=str(self.path),
            name_prefix=name_prefix,
            episode_trigger=lambda _: True,
            disable_logger=True
        )

    def reset(self, **kwargs):
        if self.video_env is not None:
            return self.video_env.reset(**kwargs)
        else:
            return self.env.reset(**kwargs)

    def step(self, action):
        if self.video_env is not None:
            return self.video_env.step(action)
        else:
            return self.env.step(action)

    def video_end(self, timeout=10.0):
        if self.video_env is None:
            return
    
        before = set(self.path.iterdir())
    
        video_env = self.video_env
        self.video_env = None
    
        video_env.close()
    
        deadline = time.monotonic() + timeout
    
        while time.monotonic() < deadline:
            files = [
                p for p in self.path.glob("*.mp4")
                if p not in before
            ]
    
            if files:
                sizes = [p.stat().st_size for p in files]
    
                time.sleep(0.1)
    
                new_sizes = [p.stat().st_size for p in files]
    
                if sizes == new_sizes and all(size > 0 for size in new_sizes):
                    return
    
            time.sleep(0.1)

    def add_evaluation(
        self,
        agent_path,
        episode,
        rewards,
        steps,
        times,
    ):
        self.evaluations.append({
            "agent": str(agent_path),
            "episode": episode,
            "mean_reward": float(np.mean(rewards)),
            "mean_steps": float(np.mean(steps)),
            "total_time": float(np.sum(times)),
            "rewards": np.asarray(rewards).tolist(),
            "steps": np.asarray(steps).tolist(),
            "times": np.asarray(times).tolist(),
        })

    def save(self):
        with open(self.path / "evaluation.json", "w") as f:
            json.dump(
                {"evaluations": self.evaluations}, f, indent=2
            )

    def close(self):
        self.video_end()
        self.save()
        self.env.close()


def iterate_agents_evaluation(env, algorithm, args, log_args=None):
    logs_path = Path(args.logspath)
    agent_models = natsorted(logs_path.glob('agents/*_model_*'), key=str)
    
    # One directory for this complete evaluation session
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    if log_args is None or 'store_path' not in log_args:
        out_path = logs_path / "eval" / session_id
    else:
        out_path = Path(log_args["store_path"]) / session_id
    
    recorder = EvaluationRecorder(env, out_path)

    for log_ep, agent_path in enumerate(agent_models):
        if args.episode > -1 and log_ep != args.episode:
            continue
        # custom agent episodes selection
        elif args.episode == -1 and log_ep not in [5, 10, 20, 35, 50]:
            continue

        print('Loading', agent_path)
        model = algorithm.load(agent_path)
        def action_selection(observations):
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
            
        if args.record:
            recorder.start_video(f"{agent_path.stem}_{log_ep:03d}")

        rewards, steps, times = evaluate_agent(
            action_selection,
            lambda : recorder.reset(),
            recorder.step,
            args.eval_steps,
            args.eval_episodes
        )

        ttime = np.sum(times).round(3)
        tsteps = np.mean(steps)
        treward = np.mean(rewards).round(4)
        sys.stdout.write(f"\r- Evaluated in {ttime:.3f} seconds | "
                         f"Mean reward: {treward:.4f} | "
                         f"Mean lenght: {tsteps}\n")
        sys.stdout.flush()
        
        # Store this checkpoint's evaluation
        recorder.add_evaluation(
            agent_path,
            log_ep,
            rewards,
            steps,
            times,
        )

        recorder.save()

    recorder.close()
