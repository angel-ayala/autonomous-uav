#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 28 00:22:19 2025

@author: angel
"""

from tqdm import tqdm
from pathlib import Path

def read_args(args_path):
    args = {}
    if '.yml' in str(args_path).lower():
        import yaml

        with open(args_path, 'r') as stream:
            try:
                args = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

    elif '.json' in str(args_path).lower():
        import json
        with open(args_path, 'r') as f:
            args = json.load(f)
    return args


def count_dirs(dir_path, n=0):
    if dir_path.exists():
        return n + 1
    else: 
        return n

logs_dir = Path('/home/angel/desarrollo/autonomous-uav/self-predictive-rl/logs/CrazyflieEnvContinuous-v0')
out_dir = Path('/home/angel/desarrollo/autonomous-uav/proprioceptive-srl/logs_cf_vector_ni')

for log_dir in tqdm(logs_dir.iterdir()):
    args_path = log_dir / 'flags.yml'
    args = read_args(args_path)
    folder_name = ''
    # append keys
    for i, k in enumerate(['agent', 'algo', 'aux', 'aux_optim', 'aux_coef']):
        if i != 0:
            folder_name += '-'
        folder_name += args[k]
    n = 1
    sym_path = (out_dir / (folder_name + f"_{n}"))
    while sym_path.exists():
        n += 1
        sym_path = (out_dir / (folder_name + f"_{n}"))
    sym_path.symlink_to(log_dir)
