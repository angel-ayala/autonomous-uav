#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 22 18:17:16 2025

@author: angel
"""

from tensorboard.backend.event_processing import event_accumulator
from pathlib import Path
import numpy as np
from scipy.interpolate import interp1d

from plot_utils import set_color_palette
from plot_utils import plot_metric
from plot_utils import draw_metric


def append_list2dict(dict_elm, key, value, multi_items=False):
    if key not in dict_elm.keys():
        dict_elm[key] = []
    if multi_items:
        dict_elm[key].extend(value)
    else:
        dict_elm[key].append(value)


def get_tb_scalar(tb_path, tb_key):
    ea = event_accumulator.EventAccumulator(
        tb_path,
        size_guidance={  # see below regarding this argument
            event_accumulator.COMPRESSED_HISTOGRAMS: 500,
            event_accumulator.IMAGES: 4,
            event_accumulator.AUDIO: 4,
            event_accumulator.SCALARS: 0,
            event_accumulator.HISTOGRAMS: 1,
            })
    ea.Reload()
    e_tags = ea.Tags()
    if tb_key in e_tags['scalars']:
        return ea.Scalars(tb_key)
    else:
        return []


def get_data_plot(epath, tb_key='train/mutual_information_zq'):
    tb_path = list(epath.glob('events*'))
    event_data = get_tb_scalar(str(tb_path[-1]), tb_key)
    # process
    steps, values = [], []
    for ed in event_data:
        steps.append(ed.step)
        values.append(ed.value)
    # format algo name
    full_name = epath.name.split('_')[0].upper()
    alg_name = full_name.replace('-JOINT', '')
    return alg_name, steps, values


base_path = Path('logs_cf_tc')

exp_paths = []
# filter
for p in base_path.iterdir():
    folder_name = p.name
    if 'random' in folder_name or 'assets' in folder_name or p.is_file():
        continue
    if 'stch' in folder_name:
        continue
    exp_paths.append(p)

exp_paths.sort(reverse=False)

# Define common steps (e.g., 450 evenly spaced steps)
common_steps = np.linspace(0, 452048, 450)

# Interpolate all lists
mutual_info = {}
for epath in exp_paths:
    algo_name, steps, values = get_data_plot(epath)
    if len(steps) == 0:
        continue
    # Create interpolation function
    interp_fn = interp1d(steps, values, kind='linear', fill_value="extrapolate")
    # Store the result
    append_list2dict(mutual_info, algo_name, interp_fn(common_steps))


with plot_metric(title='MutualInformation', ylabel='Mutual Information (MI)', xlabel='Episodes', layout='constrained', figsize=(6, 5)) as fig:
    ax = fig.add_subplot(1, 1, 1)
    set_color_palette(ax)
    for label, values in mutual_info.items():
        draw_metric(ax, label, common_steps / 9000, np.asarray(values), False)
fig.savefig(base_path / "assets/mutual_information.pdf")
fig.show()

