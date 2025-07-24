#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 20:38:44 2023

@author: Angel Ayala
"""

import time
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import threading

from webots_drone.data import ExperimentData

from plot_utils import draw_metric
from plot_utils import plot_metric
from plot_utils import plot_aggregated_metrics
from plot_utils import plot_performance_profile
from plot_utils import plot_probability_improvement
from plot_utils import plot_sample_efficiency
from plot_utils import set_color_palette


def append_list2dict(dict_elm, key, value, multi_items=False):
    if key not in dict_elm.keys():
        dict_elm[key] = []
    if multi_items:
        dict_elm[key].extend(value)
    else:
        dict_elm[key].append(value)


def append_info(exp_data, phase):
    global exp_summ
    exp_info = exp_data.get_info(phases=[phase])
    if exp_summ is None:
        exp_summ = pd.DataFrame([exp_info])
    else:
        exp_summ = pd.concat((exp_summ, pd.DataFrame([exp_info])))


def append_reward(exp_data, phase):
    global rewards
    append_list2dict(rewards, exp_data.alg_name, exp_data.get_reward_curve(phase)[:, 0])


def append_reward_quadrants(exp_data, phase):
    global rewards_tpos
    append_list2dict(rewards_tpos, exp_data.alg_name, exp_data.get_reward_curve(phase, by_quadrant=True)[:, :, 0])


def append_episodes(exp_data, phase):
    global episodes
    append_list2dict(episodes, exp_data.alg_name, exp_data.get_phase_eps(phase))


def append_metrics(exp_data, phase):
    global nav_metrics
    append_list2dict(nav_metrics, exp_data.alg_name, pd.DataFrame(exp_data.get_nav_metrics(phase)))


def process_exp_data(exp_data, phase, pbar):
    # create threads
    alive_threads = 0
    threads = []

    # try:
    threads.append(
        threading.Thread(target=append_info, args=(exp_data, phase)))
    threads.append(
        threading.Thread(target=append_reward, args=(exp_data, phase)))
    threads.append(
        threading.Thread(target=append_episodes, args=(exp_data, phase)))
    threads.append(
        threading.Thread(target=append_metrics, args=(exp_data, phase)))
    threads.append(
        threading.Thread(target=append_reward_quadrants, args=(exp_data, phase)))

    # init threads
    for thrd in threads:
        thrd.start()
        alive_threads += 1

    # monitor end and update
    while alive_threads > 0:
        for thrd in threads:
            if not thrd.is_alive():
                pbar.update()
                alive_threads -= 1
                threads.remove(thrd)
        time.sleep(0.5)

    # except AssertionError:
    #     print(exp_path, "does not have", phase, "data")

    # finally:
    return exp_data


# %% Constant definitions
# _suffix = ' (ours)'
# _ksuffix = ''
# _suffix = ''
# _ksuffix = ''
# _suffix = '/TS'
# _ksuffix = '_ts'
_suffix = '/TC'
_ksuffix = '_tc'

phase = 'eval'
exp_summ = None
methods = {}
rewards = {}
nav_metrics = {}
episodes = {}
rewards_tpos = {}

# %% Data reading and preprocessing
# base_path = Path('/home/angel/desarrollo/uav_navigation/rl_experiments/logs_cf_vector_new')
base_path = Path('/home/timevisao/angel/autonomous-uav/proprioceptive-srl/logs_cf_ispr')
# base_path = Path('/home/angel/desarrollo/autonomous-uav/proprioceptive-srl/logs_sac_ispr')
base_path = Path('/home/timevisao/angel/autonomous-uav/proprioceptive-srl/logs_cf_tc')
base_path = Path('/home/timevisao/angel/autonomous-uav/proprioceptive-srl/logs_cf_ts')
# base_path = Path('/home/angel/desarrollo/sb3-srl/logs_cf_pixel_paper')
# base_path = Path('/home/angel/desarrollo/sb3-srl/logs_cf_vector_alm')

# filter
exp_paths = []
for p in base_path.iterdir():
    folder_name = p.name
    if 'random' in folder_name or 'assets' in folder_name or p.is_file():
        continue
    # if 'alm' not in folder_name:
    #     continue
    # if 'sac' in str(p):  # filter SAC
    # if 'td3' in str(p):  # filter TD3
    # if ('spr' in folder_name or 'alm' in folder_name) and not 'ispr' in folder_name:  # filter others proposal
    # if 'stch' in str(p):  # filter Stochastic encoders
    #     continue
    exp_paths.append(p)
exp_paths.sort(reverse=False)

pbar = tqdm(total=len(exp_paths) * 5, desc="Processing")
for exp_path in exp_paths:
    pbar.set_description("Processing " + exp_path.name)
    # real_world results
    # if not (exp_path / 'eval_real').exists():
    #     continue
    # exp_data = ExperimentData(exp_path, eval_regex=r"eval_real/history_*.csv")

    # read and preprocess self-predictive-rl results
    if 'alm' in exp_path.name:  # TD3
        exp_data = ExperimentData(exp_path, exp_args='flags.yml', eval_regex=r"eval/history_*.csv")
        exp_data.exp_args['is_srl'] = True  # set as constant since is experiment-based
        # correct episode offset and filter extra data
        phase_df = exp_data.history_df[exp_data.history_df['phase'] == 'eval']
        phase_df.loc[:, 'ep'] += 1
        ep_df = phase_df[phase_df['ep'] != 51]
        ep_df = ep_df[ep_df['ep'] != 5]
        exp_data.history_df = ep_df
        exp_data.alg_name = 'TD3-Ni'
        algo_key = 'ni_et_al'
    else:
        # read normal experiment
        exp_data = ExperimentData(exp_path, eval_regex=r"eval/history_*.csv")
        algo_key = exp_data.alg_name.lower().replace('-', '_') + _ksuffix

    # proposals results
    exp_data.alg_name += _suffix
    methods[algo_key] = exp_data.alg_name
    # set (ours) in label
    # if 'ispr' in exp_path.name or 'proprio' in exp_path.name or 'rec' in exp_path.name:  # proposal
    #     exp_data.alg_name += ' (ours)'
    #     methods[algo_key] = exp_data.alg_name

    # check if already saved processed data as numpy file
    algo_data_path = base_path / f"data_{algo_key}_{phase}.npz"
    if algo_data_path.exists() and algo_data_path.is_file():
        if exp_data.alg_name in rewards.keys():
            for i in range(5):
                pbar.update()
            continue
        tmp_data = np.load(algo_data_path, allow_pickle=True)
        pbar.update()
        append_list2dict(rewards, exp_data.alg_name, tmp_data['rewards'], multi_items=True)
        pbar.update()
        nav_m = [pd.DataFrame(v) for v in tmp_data['nav_metrics']]
        append_list2dict(nav_metrics, exp_data.alg_name, nav_m, multi_items=True)
        pbar.update()
        append_list2dict(episodes, exp_data.alg_name, tmp_data['episodes'], multi_items=True)
        pbar.update()
        append_list2dict(rewards_tpos, exp_data.alg_name, tmp_data['rewards_tpos'], multi_items=True)
        pbar.update()
    else:
        # multi-thread processing
        process_exp_data(exp_data, phase, pbar=pbar)
pbar.close()

# %% Save processed data
for k, v in methods.items():
    if v not in rewards.keys():
        continue
    out_npz = base_path / f"data_{k}_{phase}.npz"
    if out_npz.exists():
        continue
    print('Saving', out_npz.name)
    tmp_dict = {
        'rewards': rewards[v],
        'nav_metrics': [m_df.to_dict() for m_df in nav_metrics[v]],
        'episodes': episodes[v],
        'rewards_tpos': rewards_tpos[v],
        }
    np.savez(out_npz, **tmp_dict)

# %% Ensure save data
out_path = base_path / 'assets'
# out_path = base_path / 'assets_poster'

# out_path = None
if out_path is not None and not out_path.exists():
    out_path.mkdir()

# %% Overall results
if out_path is not None:
    exp_summ.to_csv(out_path / 'summary.csv', index=False)


# %% Define algorithms group

algos_grp = {
    # 'baseline': ['DQN', 'SAC', 'TD3'],
    # 'DQN': ['DQN', methods['dqn_rec']],
    # 'TD3': ['TD3', methods['td3_rec']],
    # 'SAC': ['SAC'] + [methods[k] for k in ['sac_rec', 'sac_rec_stch']],
    # 'comparison': [methods[k] for k in ['dqn_rec', 'td3_rec', 'sac_rec']],

    # 'TD3':  [methods[k] for k in ['td3', 'td3_ispr', 'td3_spr']],
    # 'SAC': ['SAC'] + [methods[k] for k in ['sac', 'sac_ispr', 'sac_spr']],
    # 'comparison': [methods[k] for k in ['td3_ispr', 'sac_ispr', 'ni_et_al', 'td3_spr']],

    'baseline': [methods[k] for k in ['dqn', 'td3', 'sac' ]],
    'baseline_tc': [methods[k] for k in ['dqn_tc', 'td3_tc', 'sac_tc' ]],
    'DQN': [methods[k] for k in ['dqn', 'dqn_tc', 'dqn_ispr']],
    'TD3': [methods[k] for k in ['td3', 'td3_tc', 'td3_ispr']],
    'SAC': [methods[k] for k in ['sac', 'sac_tc', 'sac_ispr']],
    
    }

# %% Navigation metrics
nav_plots = [  # metric_id, y_label, plt_title, is_percent
    ('SR', 'Success rate', 'Success rate comparison', True),
    ('SPL', 'Success Path Length', 'Success path length comparison', True),
    # ('SSPL', 'Soft Success Path Length', 'Soft success path length comparison', True),
    ('DTS', 'Distance to success (meters)', 'Distance to success comparison', False)
]

for grp_key, grp in algos_grp.items():
    for metric_id in nav_plots:
        print('Plotting', metric_id[0])
        out_fig = out_path / f"nav_metric_{metric_id[0]}_{grp_key}.pdf" if out_path is not None else out_path
        with plot_metric(metric_id[2], metric_id[1], metric_id[-1], figsize=(7,5)) as fig:
            ax = fig.add_subplot(1, 1, 1)
            set_color_palette(ax)
            for i, (label, values) in enumerate(nav_metrics.items()):
                # print(type(values[0]), values[0].shape, metric_id[0])
                if label not in grp:
                    continue
                eps = np.asarray(episodes[label][0]) + 1
                plot_values = np.vstack([v[metric_id[0]].values for v in values])
                draw_metric(ax, label, eps, plot_values, metric_id[-1])
        if out_fig is not None:
            fig.savefig(out_fig)
        fig.show()

# %% Reward metrics
print('Plotting reward curve')
out_fig = out_path / f"reward_{phase}.pdf" if out_path is not None else out_path
with plot_metric(f"Reward curve during {phase}", 'Accumulative reward', False) as fig:
    ax = fig.add_subplot(1, 1, 1)
    set_color_palette(ax)
    for i, (label, values) in enumerate(rewards.items()):
        eps = np.asarray(episodes[label][0]) + 1
        plot_values = np.asarray(values)
        draw_metric(ax, label, eps, plot_values, False)
if out_fig is not None:
    fig.savefig(out_fig)
fig.show()

# %% Reward metrics by target quadrant
quadrant_idx = 20
print('Plotting reward curve - TargetQuadrant:', metric_id[0])
out_fig = out_path / f"reward_{phase}_tq_{quadrant_idx}.pdf" if out_path is not None else out_path
with plot_metric(f"Reward curve during {phase} for TargetQuadrant{quadrant_idx}",
                 'Accumulative reward', False) as fig:
    ax = fig.add_subplot(1, 1, 1)
    set_color_palette(ax)
    for i, (label, values) in enumerate(rewards_tpos.items()):
        eps = np.asarray(episodes[label][0]) + 1
        plot_values = np.asarray(values)[:, quadrant_idx]
        draw_metric(ax, label, eps, plot_values, False)
if out_fig is not None:
    fig.savefig(out_fig)
fig.show()

# %% Aggregated metrics with 95% Stratified Bootstrap CIs
# IQM, Optimality Gap, Median, Mean
print('Plotting aggregated reward')
algos = list(rewards_tpos.keys())
# algos = ['DQN', 'SAC', 'TD3'] + ['DQN/TC', 'SAC/TC', 'TD3/TC']
algos = [methods[k] for k in ['dqn', 'dqn_tc', 'sac', 'sac_tc', 'td3', 'td3_tc']]
algos = algos_grp['DQN'] + algos_grp['SAC'] + algos_grp['TD3']
alg_norm = 'TD3'
# alg_norm = 'SAC'
# alg_norm = ni_et_al

metric2plot = ['Median', 'IQM', 'Mean', 'Optimality Gap']
fig, axes = plot_aggregated_metrics(algos, rewards_tpos, alg_norm, metric2plot)

if out_path is not None:
    fig.savefig(out_path / "rliable_aggregated_metrics.pdf", bbox_inches='tight', pad_inches=0.1)
fig.show()

# %% Performance profile
print('Plotting Performance profile')

for grp_key, grp in algos_grp.items():
    print('Processing', grp_key)
    fig, axes = plot_performance_profile(grp, rewards_tpos, alg_norm)
    if out_path is not None:
        fig.savefig(out_path / f"rliable_performance_profile_{grp_key}.pdf", bbox_inches='tight')
    fig.show()

# %% Sample efficiency curve
print('Plotting Sample efficiency curve')

for grp_key, grp in algos_grp.items():
    fig, ax = plot_sample_efficiency(grp, rewards_tpos, alg_norm, np.asarray(episodes[alg_norm][0]))
    if out_path is not None:
        fig.savefig(out_path / f"rliable_efficiency_curve_{grp_key}.pdf", bbox_inches='tight')
    fig.show()

# %% Probability of Improvement
print('Plotting Probability of Improvement')
algos = list(rewards_tpos.keys())

algos_pairs = {
    # Target coordinates
    'baseline': [
        ('TD3', 'SAC'),
        ('DQN', 'SAC'),
        ('DQN', 'TD3'),
        ],
    'SAC': [
        (methods['sac_rec'], methods['td3_rec']),
        (methods['sac_rec'], methods['dqn_rec']),
        (methods['sac_rec'], 'SAC'),
        (methods['sac_rec'], 'TD3'),
        (methods['sac_rec'], 'DQN'),
        ],
    'TD3': [
        (methods['td3_rec'], methods['sac_rec']),
        (methods['td3_rec'], methods['dqn_rec']),
        (methods['td3_rec'], 'SAC'),
        (methods['td3_rec'], 'TD3'),
        (methods['td3_rec'], 'DQN'),
        ],
    'DQN': [
        (methods['dqn_rec'], methods['sac_rec']),
        (methods['dqn_rec'], methods['td3_rec']),
        (methods['dqn_rec'], 'SAC'),
        (methods['dqn_rec'], 'TD3'),
        (methods['dqn_rec'], 'DQN'),
        ],
    'reconstruction': [
        (methods['td3_rec'], methods['sac_rec']),
        (methods['dqn_rec'], methods['sac_rec']),
        (methods['dqn_rec'], methods['td3_rec']),
        (methods['sac_rec'], 'SAC'),
        (methods['td3_rec'], 'TD3'),
        (methods['dqn_rec'], 'DQN'),
        ],
}
algos_pairs = {
    # Target coordinates
    'baseline': [
        ('TD3', 'SAC'),
        ('DQN', 'SAC'),
        ('DQN', 'TD3'),
        ],
    }
    # 'TD3': [
    #     (ispr_td3_key, 'TD3'),
    #     (proprio_td3_key, 'TD3'),
    #     (proprio_td3_key, ispr_td3_key),
    #     ],
    # 'SAC': [
    #     (ispr_sac_key, 'SAC'),
    #     (proprio_sac_key, 'SAC'),
    #     (proprio_sac_key, ispr_sac_key),
    #     ],
    # 'self-predictive': [
    #     # cross methods
    #     (ispr_sac_key, ispr_td3_key),
    #     ('SAC', ispr_td3_key),
    #     ('SAC', ispr_sac_key),
    #     ('TD3', ispr_td3_key),
    #     ('TD3', ispr_sac_key),
    #     ('TD3',   'SAC'),
    #     ],
    # 'sota': [
    #     (ispr_td3_key,  ni_et_al),
    #     (ispr_sac_key,  ni_et_al),
    #     (ispr_td3_key,  ispr_sac_key),
    #     ('TD3-SPR', 'SAC-SPR'),
    #     ('TD3',   'SAC'),
    #     ],
    # Proprioceptive
    # Baselines

for pair_key, pair in algos_pairs.items():
    print('Processing', pair_key)
    fig, axes = plot_probability_improvement(algos, rewards_tpos, alg_norm, pair)
    if out_path is not None:
        fig.savefig(out_path / f"rliable_probability_improvement_{pair_key}.pdf", bbox_inches='tight')
    fig.show()

