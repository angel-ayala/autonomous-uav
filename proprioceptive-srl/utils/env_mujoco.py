#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 19 16:32:40 2025

@author: angel
"""
from typing import Any, Callable, Union, Optional
import os
import warnings
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.core import (
    ActType,
    ObsType,
    WrapperObsType
    )


from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, sync_envs_normalization


def get_env(env_id: str, seed: int = 666):
    env = gym.make(env_id)

    if env.observation_space.dtype == np.float64:
        env = DtypeObservation(env, np.float32)

    env = gym.wrappers.RecordEpisodeStatistics(env)
    env.seed(seed=seed)
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


# class CustomEvalCallback(EvalCallback):
#     """
#     Callback for evaluating an agent.

#     .. warning::

#       When using multiple environments, each call to  ``env.step()``
#       will effectively correspond to ``n_envs`` steps.
#       To account for that, you can use ``eval_freq = max(eval_freq // n_envs, 1)``

#     :param eval_env: The environment used for initialization
#     :param callback_on_new_best: Callback to trigger
#         when there is a new best model according to the ``mean_reward``
#     :param callback_after_eval: Callback to trigger after every evaluation
#     :param n_eval_episodes: The number of episodes to test the agent
#     :param eval_freq: Evaluate the agent every ``eval_freq`` call of the callback.
#     :param log_path: Path to a folder where the evaluations (``evaluations.npz``)
#         will be saved. It will be updated at each evaluation.
#     :param best_model_save_path: Path to a folder where the best model
#         according to performance on the eval env will be saved.
#     :param deterministic: Whether the evaluation should
#         use a stochastic or deterministic actions.
#     :param render: Whether to render or not the environment during evaluation
#     :param verbose: Verbosity level: 0 for no output, 1 for indicating information about evaluation results
#     :param warn: Passed to ``evaluate_policy`` (warns if ``eval_env`` has not been
#         wrapped with a Monitor wrapper)
#     """

#     def __init__(
#         self,
#         eval_env: Union[gym.Env, VecEnv],
#         callback_on_new_best: Optional[BaseCallback] = None,
#         callback_after_eval: Optional[BaseCallback] = None,
#         n_eval_episodes: int = 5,
#         eval_freq: int = 10000,
#         log_path: Optional[str] = None,
#         best_model_save_path: Optional[str] = None,
#         deterministic: bool = True,
#         render: bool = False,
#         verbose: int = 1,
#         warn: bool = True,
#     ):
#         super().__init__(callback_after_eval, verbose=verbose)

#         self.callback_on_new_best = callback_on_new_best
#         if self.callback_on_new_best is not None:
#             # Give access to the parent
#             self.callback_on_new_best.parent = self

#         self.n_eval_episodes = n_eval_episodes
#         self.eval_freq = eval_freq
#         self.best_mean_reward = -np.inf
#         self.last_mean_reward = -np.inf
#         self.deterministic = deterministic
#         self.render = render
#         self.warn = warn

#         # Convert to VecEnv for consistency
#         if not isinstance(eval_env, VecEnv):
#             eval_env = DummyVecEnv([lambda: eval_env])  # type: ignore[list-item, return-value]

#         self.eval_env = eval_env
#         self.best_model_save_path = best_model_save_path
#         # Logs will be written in ``evaluations.npz``
#         if log_path is not None:
#             log_path = os.path.join(log_path, "evaluations")
#         self.log_path = log_path
#         self.evaluations_results: list[list[float]] = []
#         self.evaluations_timesteps: list[int] = []
#         self.evaluations_length: list[list[int]] = []
#         # For computing success rate
#         self._is_success_buffer: list[bool] = []
#         self.evaluations_successes: list[list[bool]] = []

#     def _init_callback(self) -> None:
#         # Does not work in some corner cases, where the wrapper is not the same
#         if not isinstance(self.training_env, type(self.eval_env)):
#             warnings.warn("Training and eval env are not of the same type" f"{self.training_env} != {self.eval_env}")

#         # Create folders if needed
#         if self.best_model_save_path is not None:
#             os.makedirs(self.best_model_save_path, exist_ok=True)
#         if self.log_path is not None:
#             os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

#         # Init callback called on new best model
#         if self.callback_on_new_best is not None:
#             self.callback_on_new_best.init_callback(self.model)

#     def _log_success_callback(self, locals_: dict[str, Any], globals_: dict[str, Any]) -> None:
#         """
#         Callback passed to the  ``evaluate_policy`` function
#         in order to log the success rate (when applicable),
#         for instance when using HER.

#         :param locals_:
#         :param globals_:
#         """
#         info = locals_["info"]

#         if locals_["done"]:
#             maybe_is_success = info.get("is_success")
#             if maybe_is_success is not None:
#                 self._is_success_buffer.append(maybe_is_success)

#     # def _eval(self):
#     #     returns = 0
#     #     steps = 0
#     #     # Reset success rate buffer
#     #     self._is_success_buffer = []
#     #     for _ in range(self.n_eval_episodes):
#     #         done = False
#     #         state = self.eval_env.reset()
#     #         # TODO: action selection function
#     #         while not done:
#     #             action = self.agent.get_action(state, self._train_step, eval=True)
#     #             next_state, _, done, info = self.eval_env.step(action)
#     #             state = next_state

#     #         returns += info["episode"]["r"]
#     #         steps += info["episode"]["l"]

#     #         print(
#     #             "EVAL Episode: {}, total numsteps: {}, return: {}".format(
#     #                 self._train_episode,
#     #                 self.n_calls,
#     #                 round(info["episode"]["r"], 2),
#     #             )
#     #         )

#     #     eval_metrics = dict()
#     #     eval_metrics["return"] = returns / self.n_eval_episodes
#     #     eval_metrics["length"] = steps / self.n_eval_episodes

#     #     if (
#     #         self.best_model_save_path is not None
#     #         and returns / self.n_eval_episodes >= self._best_eval_returns
#     #     ):
#     #         self.save_snapshot(best=True)
#     #         self._best_eval_returns = returns / self.n_eval_episodes

#     #     # logger.record_step("env_steps", self._train_step)
#     #     # for k, v in eval_metrics.items():
#     #     #     logger.record_tabular(k, v)
#     #     # logger.dump_tabular()

#     def _on_step(self) -> bool:
#         continue_training = True

#         if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
#             # Sync training and eval env if there is VecNormalize
#             if self.model.get_vec_normalize_env() is not None:
#                 try:
#                     sync_envs_normalization(self.training_env, self.eval_env)
#                 except AttributeError as e:
#                     raise AssertionError(
#                         "Training and eval env are not wrapped the same way, "
#                         "see https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html#evalcallback "
#                         "and warning above."
#                     ) from e

#             # Reset success rate buffer
#             self._is_success_buffer = []

#             episode_rewards, episode_lengths = evaluate_policy(
#                 self.model,
#                 self.eval_env,
#                 n_eval_episodes=self.n_eval_episodes,
#                 render=self.render,
#                 deterministic=self.deterministic,
#                 return_episode_rewards=True,
#                 warn=self.warn,
#                 callback=self._log_success_callback,
#             )

#             if self.log_path is not None:
#                 assert isinstance(episode_rewards, list)
#                 assert isinstance(episode_lengths, list)
#                 self.evaluations_timesteps.append(self.num_timesteps)
#                 self.evaluations_results.append(episode_rewards)
#                 self.evaluations_length.append(episode_lengths)

#                 kwargs = {}
#                 # Save success log if present
#                 if len(self._is_success_buffer) > 0:
#                     self.evaluations_successes.append(self._is_success_buffer)
#                     kwargs = dict(successes=self.evaluations_successes)

#                 np.savez(
#                     self.log_path,
#                     timesteps=self.evaluations_timesteps,
#                     results=self.evaluations_results,
#                     ep_lengths=self.evaluations_length,
#                     **kwargs,  # type: ignore[arg-type]
#                 )

#             mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
#             mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
#             self.last_mean_reward = float(mean_reward)

#             if self.verbose >= 1:
#                 print(f"Eval num_timesteps={self.num_timesteps}, " f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
#                 print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")
#             # Add to current Logger
#             self.logger.record("eval/mean_reward", float(mean_reward))
#             self.logger.record("eval/mean_ep_length", mean_ep_length)

#             if len(self._is_success_buffer) > 0:
#                 success_rate = np.mean(self._is_success_buffer)
#                 if self.verbose >= 1:
#                     print(f"Success rate: {100 * success_rate:.2f}%")
#                 self.logger.record("eval/success_rate", success_rate)

#             # Dump log so the evaluation results are printed with the correct timestep
#             self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
#             self.logger.dump(self.num_timesteps)

#             if mean_reward > self.best_mean_reward:
#                 if self.verbose >= 1:
#                     print("New best mean reward!")
#                 if self.best_model_save_path is not None:
#                     self.model.save(os.path.join(self.best_model_save_path, "best_model"))
#                 self.best_mean_reward = float(mean_reward)
#                 # Trigger callback on new best model, if needed
#                 if self.callback_on_new_best is not None:
#                     continue_training = self.callback_on_new_best.on_step()

#             # Trigger callback after every evaluation, if needed
#             if self.callback is not None:
#                 continue_training = continue_training and self._on_event()

#         return continue_training

#     def update_child_locals(self, locals_: dict[str, Any]) -> None:
#         """
#         Update the references to the local variables.

#         :param locals_: the local variables during rollout collection
#         """
#         if self.callback:
#             self.callback.update_locals(locals_)
