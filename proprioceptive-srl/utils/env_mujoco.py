#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 19 16:32:40 2025

@author: angel
"""
from typing import Any, Callable, Optional, List
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.core import (
    ActType,
    ObsType,
    WrapperObsType
    )


from stable_baselines3.common.utils import get_latest_run_id
from stable_baselines3.common.callbacks import EvalCallback

from .agent import save_dict_json


def get_env(env_id: str, seed: int = 666):
    env = gym.make(env_id)

    if env.observation_space.dtype == np.float64:
        env = DtypeObservation(env, np.float32)

    env = gym.wrappers.RecordEpisodeStatistics(env)
    # env.seed(seed=seed)
    env.observation_space.seed(seed)
    env.action_space.seed(seed)
    return env


def parse_training_args(parser):
    arg_training = parser.add_argument_group('Training')
    arg_training.add_argument("--environment-id", type=str, default="Ant-v2",
                              help='The Mujoco control environment name ID.')
    arg_training.add_argument("--steps", type=int, default=500000,  # 25h at 25 frames
                              help='Number of training steps.')
    arg_training.add_argument('--memory-steps', type=int, default=5000,
                              help='Number of steps for initial population of the Experience replay buffer.')
    arg_training.add_argument("--batch-size", type=int, default=512,
                              help='Minibatch size for training.')
    arg_training.add_argument('--eval-interval', type=int, default=10000,  # 30m at 25 frames
                              help='Steps interval for progress evaluation.')
    arg_training.add_argument('--eval-episodes', type=int, default=10,  # 1m at 25 frames
                              help='Number of evaluation steps.')
    return arg_training


def args2logpath(args, algo, env_uav='cf'):
    if args.logspath is None:
        # Summary folder
        outfolder = f"logs_{env_uav}"
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
    exp_name = f"{args.environment_id}-{algo}{path_suffix}"

    latest_run_id = get_latest_run_id(outfolder, exp_name)

    return outfolder, exp_name, latest_run_id


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


def env_id2prop_mask(env_id: str) -> Optional[List]:
    prop_mask = None
    env_lower = env_id.lower()

    if "ant" in env_lower:
        prop_mask = ANT
    elif "cheeta" in env_lower:
        prop_mask = CHEETAH
    elif "hopper" in env_lower:
        prop_mask = HOPPER

    return prop_mask

# Constant definitions of boolean proprioceptive masks for MuJoCo environments.
# True = proprioceptive information, False = Exteroceptive information


ANT = [
    False,  # 0: z-coordinate of torso
    False,  # 1: w-orientation
    False,  # 2: x-orientation
    False,  # 3: y-orientation
    False,  # 4: z-orientation
    True,   # 5: hip_1 angle
    True,   # 6: ankle_1 angle
    True,   # 7: hip_2 angle
    True,   # 8: ankle_2 angle
    True,   # 9: hip_3 angle
    True,   # 10: ankle_3 angle
    True,   # 11: hip_4 angle
    True,   # 12: ankle_4 angle
    False,  # 13: x-velocity
    False,  # 14: y-velocity
    False,  # 15: z-velocity
    False,  # 16: x-angular velocity
    False,  # 17: y-angular velocity
    False,  # 18: z-angular velocity
    True,   # 19: hip_1 angular velocity
    True,   # 20: ankle_1 angular velocity
    True,   # 21: hip_2 angular velocity
    True,   # 22: ankle_2 angular velocity
    True,   # 23: hip_3 angular velocity
    True,   # 24: ankle_3 angular velocity
    True,   # 25: hip_4 angular velocity
    True,   # 26: ankle_4 angular velocity
]

CHEETAH = [
    False,  # 0: z-coordinate of front tip (absolute position)
    False,  # 1: angle of front tip (orientation, requires external reference)
    True,   # 2: angle of back thigh (joint angle)
    True,   # 3: angle of back shin (joint angle)
    True,   # 4: angle of back foot (joint angle)
    True,   # 5: angle of front thigh (joint angle)
    True,   # 6: angle of front shin (joint angle)
    True,   # 7: angle of front foot (joint angle)
    False,  # 8: velocity of x-coordinate of front tip (requires external reference)
    False,  # 9: velocity of z-coordinate of front tip (requires external reference)
    False,  # 10: angular velocity of front tip (requires external reference)
    True,   # 11: angular velocity of back thigh (joint angular velocity)
    True,   # 12: angular velocity of back shin (joint angular velocity)
    True,   # 13: angular velocity of back foot (joint angular velocity)
    True,   # 14: angular velocity of front thigh (joint angular velocity)
    True,   # 15: angular velocity of front shin (joint angular velocity)
    True,   # 16: angular velocity of front foot (joint angular velocity)
]

HOPPER = [
    False,  # 0: z-coordinate of torso (absolute position)
    False,  # 1: angle of torso (orientation, requires external reference)
    True,   # 2: angle of thigh joint (joint angle)
    True,   # 3: angle of leg joint (joint angle)
    True,   # 4: angle of foot joint (joint angle)
    False,  # 5: velocity of x-coordinate of torso (requires external reference)
    False,  # 6: velocity of z-coordinate of torso (requires external reference)
    False,  # 7: angular velocity of torso (requires external reference)
    True,   # 8: angular velocity of thigh joint (joint angular velocity)
    True,   # 9: angular velocity of leg joint (joint angular velocity)
    True,   # 10: angular velocity of foot joint (joint angular velocity)
]

