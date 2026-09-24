#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 11:28:26 2026

@author: angel
"""
from typing import Callable

import time
import sys


def evaluate_agent(agent_policy: Callable,
                   env_reset: Callable,
                   env_step: Callable,
                   n_steps: int,
                   n_episodes: int = 1):
    steps = []
    rewards = []
    times = []

    for i in range(n_episodes):
        timemark = time.time()
        state, info = env_reset()
        ep_reward = 0
        ep_steps = 0
        end = False

        while not end:
            action = agent_policy(state)
            next_state, reward, done, truncated, info = env_step(action)
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

    return rewards, steps, times
