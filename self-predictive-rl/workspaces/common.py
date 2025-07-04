from pathlib import Path
import numpy as np
from utils import logger


def make_agent(env, device, cfg):
    if cfg.agent == "alm":
        from agents.alm import AlmAgent

        num_states = np.prod(env.observation_space.shape)
        num_actions = np.prod(env.action_space.shape)
        action_low = env.action_space.low[0]
        action_high = env.action_space.high[0]

        if cfg.id == "Humanoid-v2":
            cfg.env_buffer_size = 1000000
        buffer_size = min(cfg.env_buffer_size, cfg.num_train_steps)

        agent = AlmAgent(
            device,
            action_low,
            action_high,
            num_states,
            num_actions,
            buffer_size,
            cfg,
        )

    else:
        raise NotImplementedError

    return agent


def make_env(cfg):
    if cfg.benchmark == "gym-drone":
        from utils.misc import args2env_params, instance_env

        # Environment
        environment_name = 'webots_drone:webots_drone/CrazyflieEnvContinuous-v0'

        def get_env(cfg):
            env_params = args2env_params(cfg)
            env = instance_env(environment_name, env_params, seed=cfg.seed)
            env.observation_space.seed(cfg.seed)
            env.action_space.seed(cfg.seed)
            logger.log(env.observation_space.shape, env.action_space)
            return env

        return get_env(cfg)#, get_env(cfg)

    else:
        raise NotImplementedError
