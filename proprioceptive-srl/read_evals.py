#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 20 22:03:02 2025

@author: angel
"""

from pathlib import Path
import numpy as np
import pandas as pd
import copy

from plot_utils import plot_aggregated_metrics
from plot_utils import plot_sample_efficiency
from plot_utils import plot_performance_profile
from plot_utils import plot_probability_improvement


def append_list2dict(dict_elm, key, value):
    if key not in dict_elm.keys():
        dict_elm[key] = value
    else:
        dict_elm[key] = np.vstack((dict_elm[key], value))


def path2rewards(logs_path: Path):
    # process SB3 output
    eval_path = logs_path / 'evaluations/evaluations.npz'
    if eval_path.exists():
        eval_npy = np.load(eval_path)
        return eval_npy['results'].transpose((1, 0))[np.newaxis, ]
    # process td3-ni
    eval_path = logs_path / 'eval/progress.csv'
    if eval_path.exists():
        progress_df = pd.read_csv(eval_path)
        # Group by 'env_steps' and aggregate 'return' values
        aggregated = progress_df.groupby('env_steps')['return'].agg(list).reset_index()
        # Convert the list of returns into a NumPy array
        result_array = np.array(aggregated['return'].tolist() + [aggregated['return'].tolist()[-1]])
        return result_array.transpose((1, 0))[np.newaxis, ]
    return None


def path2name(logs_path: str):
    logs_path = str(logs_path.name)
    if 'ours-' in logs_path or 'alm-' in logs_path:
        return 'TD3-Ni'

    algo_name = logs_path.split('_')[0]
    algo_name = algo_name.replace('Ant-v2-', '')
    algo_name = algo_name.replace('Ant-v4-', '')
    algo_name = algo_name.replace('-joint', '')
    # algo_name = algo_name.replace('-proprio', '-pespr')
    algo_name = algo_name.upper()


    # proposals results

    if 'ispr' in logs_path:
        algo_name = algo_name.replace('ISPR', 'AmelPred')

    if 'proprio' in logs_path:
        algo_name = algo_name.replace('PROPRIO', 'AmelPred-APE')

    if 'rstch' in logs_path:
        algo_name = algo_name.replace('-RSTCH', '') + 'Sto'
    elif 'stch' in logs_path:
        algo_name = algo_name.replace('-STCH', '') + 'Sto'
    elif 'rdet' in logs_path:
        algo_name = algo_name.replace('-RDET', '') + 'Det'
    else:
        algo_name += 'Det'
    
    # variants
    if '-gradn100.0' in logs_path:
        algo_name = algo_name.replace('-GRADN100.0', '') + '_GradNorm'
        
    if '-blnc2' in logs_path:
        algo_name = algo_name.replace('-BLNC2', '') + '_Balanced2digits'
    elif '-blnc' in logs_path:
        algo_name = algo_name.replace('-BLNC', '') + '_Balanced6digits'

    if '-eb' in logs_path:
        algo_name = algo_name.replace('-eb', '') + '_EntropyBeta'
    
    # latent-dim    
    if '-ltn64' in logs_path:
        algo_name = algo_name.replace('-LTN64', '') + '_Latent64'
    
    if '-ltn128' in logs_path:
        algo_name = algo_name.replace('-LTN128', '') + '_Latent128'
    
    if '-ltn256' in logs_path:
        algo_name = algo_name.replace('-LTN256', '') + '_Latent256'
    
    if '-ltn512' in logs_path:
        algo_name = algo_name.replace('-LTN512', '') + '_Latent512'
    
    # feature-dim    
    if '-feat64' in logs_path:
        algo_name = algo_name.replace('-FEAT64', '') + '_Feat64'
    
    if '-feat128' in logs_path:
        algo_name = algo_name.replace('-FEAT128', '') + '_Feat128'
    
    if '-feat256' in logs_path:
        algo_name = algo_name.replace('-FEAT256', '') + '_Feat256'
    
    if '-feat512' in logs_path:
        algo_name = algo_name.replace('-FEAT512', '') + '_Feat512'
    
    # fusion labels
    if '-fmlp' in logs_path:
        algo_name = algo_name.replace('-FMLP', '') + '_FusionMLP'

    if '-fconv1dv0' in logs_path:
        algo_name = algo_name.replace('-FCONV1DV0', '') + '_FusionConv1Dv0'

    elif '-fconv1dv1' in logs_path:
        algo_name = algo_name.replace('-FCONV1DV1', '') + '_FusionConv1Dv1'
    
    elif '-fconv1dv2' in logs_path:
        algo_name = algo_name.replace('-FCONV1DV2', '') + '_FusionConv1Dv2'

    elif '-fconv1d' in logs_path:
        algo_name = algo_name.replace('-FCONV1D', '') + '_FusionConv1D'

    elif '-ffilm' in logs_path:
        algo_name = algo_name.replace('-FFILM', '') + '_FusionFiLM'

    elif '-fgated' in logs_path:
        algo_name = algo_name.replace('-FGATED', '') + '_FusionGated'

    if '_late' in logs_path:
        algo_name += '_Late'

    return algo_name


def path2args(logs_path: Path):
    # process SB3 output
    json_path = logs_path / 'arguments.json'
    if json_path.exists():
        import json
        with open(json_path, 'r') as json_file:
            args = json.load(json_file)
        return args
    # process td3-ni
    yaml_path = logs_path / 'flags.yml'
    if yaml_path.exists():
        import yaml
        with open(yaml_path, 'r') as file:
            args = yaml.safe_load(file)
        # replace to json defaults
        args['encoder_tau'] = args['tau']
        args['latent_dim'] = args['latent_dims']
        args['hidden_dim'] = args['hidden_dims']
        args['num_layers'] = 2
        args['model_hidden_dim'] = args['model_hidden_dims']
        tmp_lr = args['lr']
        args['lr'] = tmp_lr['critic']
        args['encoder_lr'] = tmp_lr['encoder']
        args['decoder_lr'] = tmp_lr['model']
        return args
    return None


def args2name(exp_path):
    algo_name = path2name(exp_path.stem)
    algo_name = algo_name.replace("**", "")

    args = path2args(exp_path)
    keys = [
        'encoder_tau',
        'latent_dim',
        'hidden_dim',
        'num_layers',
        'model_hidden_dim',
        'lr',
        'encoder_lr',
        'decoder_lr'
        ]

    params_kv = []
    for k in keys:
        # scientific notation
        if 'tau' in k:
            _suffix = f"{args[k]:.3f}"
        # floating point
        elif 'lr' in k:
            _suffix = f"{args[k]:.0e}"
        else:
            _suffix = f"{args[k]}"

        _prefix = "".join(x[0] for x in k.split("_"))
        params_kv.append(_prefix + _suffix)

    suffix = "_".join(params_kv)
    base_lr = False
    base_size = False
    base_tau = False
    is_ni_et_al = 'TD3-Ni' == algo_name

    sizes_dict = {
        'ld32_hd512_nl1_mhd256':  "_size_00",
        'ld32_hd1024_nl1_mhd512':  "_size_01",
        'ld32_hd512_nl1_mhd1024':  "_size_02",
        'ld32_hd1024_nl1_mhd1024': "_size_03",
        'ld50_hd512_nl1_mhd256':   "_size_04",
        'ld50_hd512_nl1_mhd1024':  "_size_05",
        'ld50_hd1024_nl1_mhd512':  "_size_06",
        'ld50_hd1024_nl1_mhd1024': "_size_07",
        'ld50_hd512_nl2_mhd1024':  "_size_08",
        'ld50_hd1024_nl2_mhd512':  "_size_09",
        'ld50_hd1024_nl2_mhd1024': "_size_10",
        }

    lrs_dict = {
        'l1e-03_el1e-03_dl1e-03': "_lr_1e-03->1e-03",
        'l3e-04_el1e-03_dl1e-03': "_lr_3e-04->1e-03",
        'l1e-04_el1e-04_dl1e-04': "_lr_1e-04->1e-04",
        'l3e-04_el3e-04_dl3e-04': "_lr_3e-04->3e-04",
        'l1e-04_el5e-05_dl1e-04': "_lr_5e-05->1e-04",
        'l3e-04_el5e-05_dl3e-04': "_lr_5e-05->3e-04",
        'l5e-05_el1e-04_dl1e-04': "out", #"_lr_1e-04->5e-05",
        }

    # if 'TD3' in algo_name:
    #     if 'l1e-03_el1e-03_dl1e-03' in suffix:
    #         base_lr = True

    # if 'SAC' in algo_name:
    #     if 'l3e-04_el1e-03_dl1e-03' in suffix:
    #         base_lr = True

    # if 'ld32_hd512_nl1_mhd256' in suffix:
    #     base_size = True

    # if 'et0.999' in suffix:
    #     base_tau = True

    # if 'SAC' in algo_name and suffix == 'et0.900_ld50_hd1024_nl1_mhd512_l3e-04_el3e-04_dl3e-04':
    #     algo_name += "_best"
    # elif 'TD3' in algo_name and suffix == 'et0.900_ld50_hd1024_nl2_mhd1024_l1e-04_el5e-05_dl1e-04':
    #     algo_name += "_best"
    if not is_ni_et_al:
        # first iterate for LR label
        # if not base_lr:
        for k, v in lrs_dict.items():
            if k in suffix:
                algo_name += v

        # first iterate for tau label
        # if not base_tau:
        algo_name += "_tau_" + suffix.split("_")[0].replace('et', '')

        # first iterate for size label
        # if not base_size:  # and base_lr and base_tau:
        for k, v in sizes_dict.items():
            if k in suffix:
                algo_name += v

    return algo_name


def append_path(exp_path):
    paths = list(exp_path.iterdir())
    paths = sorted(paths)
    for exp_path in paths:
        if not exp_path.is_dir():
            continue
        if 'Ant-v2' in exp_path.name:
            continue
        if 'assets' in exp_path.name:
            continue

        algo_name = path2name(exp_path)
        # algo_name = args2name(exp_path)
        if 'out' in algo_name:
            continue

        append_list2dict(rewards_eval,
                         algo_name,
                         path2rewards(exp_path))


def append2list_create(element, dict_elm, dict_key):
    if dict_key not in dict_elm.keys():
        dict_elm[dict_key] = []
    dict_elm[dict_key].append(element)

rewards_eval = {}
# mix faltantes
# SAC 3e-4_3e-4 tau 0.9 size_1, 4, 9
# TD3 3e-4_3e-4 tau 0.9 size_1, 4, 6, 9

# TD3 5e-5_3e-4 tau 0.9, 0.99 size_10

# TD3 5e-5_1e-4 tau 0.99 size_1, 4, 6, 9, 10

# %% append folder
# leer recompensas

paths = [
    # Path('/home/timevisao/angel/autonomous-uav/proprioceptive-srl'
    #      '/logs_regularization/Ant-v4/'),
    Path('/run/user/1000/gvfs/sftp:host=10.80.1.99,user=timevisao'
         '/home/timevisao/angel/autonomous-uav/proprioceptive-srl/logs/Ant-v4'),
    Path('/run/user/1000/gvfs/sftp:host=10.80.1.97'
         '/home/timevisao/angel/autonomous-uav/proprioceptive-srl/logs/Humanoid-v4')
    ]
# base_path = Path('/home/timevisao/angel/autonomous-uav/proprioceptive-srl'
#                  '/logs_mujoco_ablation')
#                  '/logs_mujoco')

# paths = [
#     Path('/mnt/storage/angel/mujoco_ablation'),
#     Path('/mnt/storage/angel/mujoco_ablation_part2'),
#     Path('/mnt/storage/angel/mujoco_ablation_part3'),
#     Path('/mnt/storage/angel/mujoco_ablation_part4'),
#     Path('/mnt/storage/angel/mujoco_ablation_part5'),
#     Path('/mnt/storage/angel/mujoco_ablation_part6'),
#     Path('/mnt/storage/angel/mujoco_ablation_part7'),
#     Path('/mnt/storage/angel/mujoco_ablation_part8'),
#     Path('/mnt/storage/angel/mujoco_ablation_part9'),
#     Path('/mnt/storage/angel/mujoco_ablation_part10')
#     ]

for p in paths:
    append_path(p)

algos_keys = list(rewards_eval.keys())
algos_keys.sort()
algos_keys

# %% Create out path
out_path = paths[-1].parent / 'assets_v2'
if out_path:
    out_path.mkdir(exist_ok=True)

# %% Define algorithms group
algos_grp = {
    'comparison': ['TD3-Ni', 'SAC-AmelPredSto**', 'TD3-AmelPredSto**'],
    'SAC': ['SAC-AmelPredSto', 'SAC-AmelPredSto*',  'SAC-AmelPredSto**'],
    'TD3': ['TD3-AmelPredSto', 'TD3-AmelPredSto*',  'TD3-AmelPredSto**'],
    }
algos_grp['all'] = ['TD3-Ni'] + algos_grp['SAC'] + algos_grp['TD3']
# %% Define algorithms group
alg_norm = 'TD3-AmelPredDet'

algos_grp = {
    'regDet' : [
        'TD3-AmelPredDet',
        'TD3-AmelPredSto',
        'TD3-AmelPredDet_Balanced2digits',
        'TD3-AmelPredDet_Balanced6digits',
        'TD3-AmelPredDet_GradNorm',
        'TD3-AmelPredDet_GradNorm_Balanced6digits',
    ],
    'regDetSac' : [
        'TD3-AmelPredDet',
        'SAC-AmelPredDet',
        'SAC-AmelPredDet_Balanced6digits',
        'SAC-AmelPredDet_GradNorm',
        'SAC-AmelPredDet_GradNorm_Balanced6digits',
    ],
    'regSto' : [
        'TD3-AmelPredDet',
        'TD3-AmelPredSto',
        'TD3-AmelPredSto_Balanced2digits',
        'TD3-AmelPredSto_Balanced6digits',
        'TD3-AmelPredSto_GradNorm',
        'TD3-AmelPredSto_GradNorm_Balanced6digits'
    ],
    'regStoSac' : [
        'TD3-AmelPredDet',
        'SAC-AmelPredSto',
        'SAC-AmelPredSto_Balanced6digits',
        'SAC-AmelPredSto_GradNorm',
        'SAC-AmelPredSto_GradNorm_Balanced6digits',
    ],
    'featDet': [
        'TD3-AmelPredDet',
        'TD3-AmelPredDet_Feat64',
        'TD3-AmelPredDet_Feat128',
        'TD3-AmelPredDet_Feat256',
        'TD3-AmelPredDet_Feat512',
     ],
    'featSto': [
        'TD3-AmelPredDet',
        'TD3-AmelPredSto',
        'TD3-AmelPredSto_Feat64',
        'TD3-AmelPredSto_Feat128',
        'TD3-AmelPredSto_Feat256',
        'TD3-AmelPredSto_Feat512',
    ],
   'latentDet': [
       'TD3-AmelPredDet',
       'TD3-AmelPredDet_Latent64',
       'TD3-AmelPredDet_Latent128',
       'TD3-AmelPredDet_Latent256',
       'TD3-AmelPredDet_Latent512',
   ],
   'latentSto': [
       'TD3-AmelPredDet',
       'TD3-AmelPredSto',
       'TD3-AmelPredSto_Latent64',
       'TD3-AmelPredSto_Latent128',
       'TD3-AmelPredSto_Latent256',
       'TD3-AmelPredSto_Latent512',
   ],
}

# %% Define algorithms group
algos_grp = {'all': algos_keys}

# %% Define algorithms group

algos_grp = {}
for algo_key in algos_keys:
    algo_base = algo_key.split('-')[0]
    grp_key = f"{algo_base}_"
    look_keys = ['tau_', 'size_', 'lr_']
    have_keys = [k in algo_key for k in look_keys]
    look_values = [algo_key.split(k)[-1].split('_')[0] for k in look_keys]
    settings = []
    n_keys = sum(have_keys)

    # if n_keys == 0:
    #     grp_key += "baseline"
    #     append2list_create(algo_key, algos_grp, grp_key)
    # else:
    if n_keys > 0:

        for i, (key, presence, value) in enumerate(
                zip(look_keys, have_keys, look_values)):
            if presence:
                settings.append(key[:-1])
                settings.append(f"{key}{value}")

        # grp_key = "_".join(settings)
        # if n_keys == 1:
        #     grp_key += "only"


        # same hyperparameter group
        for s in settings:
            append2list_create(algo_key, algos_grp, grp_key + s)

# for k, v in algos_grp.items():
#     algo_base = k.split("_")[0]
#     algos_grp[k].append(algo_base + "-AmelPredSto")

# %% Groups 2
algos_grp = {
    # 'comparison': ['TD3-Ni', 'SAC-ISPR-STCH**', 'TD3-ISPR-STCH**'],
    # 'SAC': ['SAC-ISPR-STCH', 'SAC-ISPR-STCH*',  'SAC-ISPR-STCH**'],
    # 'TD3': ['TD3-ISPR-STCH', 'TD3-ISPR-STCH*',  'TD3-ISPR-STCH**'],
    'SAC': [a for a in algos_keys if 'SAC' in a],
    'TD3': [a for a in algos_keys if 'TD3' in a and '-Ni' not in a],
    }
algos_grp['SAC_tau'] = ['SAC-AmelPredSto']
algos_grp['SAC_size'] = ['SAC-AmelPredSto']
algos_grp['SAC_lr'] = ['SAC-AmelPredSto']
algos_grp['SAC_combo'] = ['SAC-AmelPredSto']

for algo in algos_grp['SAC']:
    is_tau = '_tau_' in algo
    is_size = '_size_' in algo
    is_lr = '_lr_' in algo

    if is_tau and not is_size and not is_lr:
        algos_grp['SAC_tau'].append(algo)
    elif not is_tau and is_size and not is_lr:
        algos_grp['SAC_size'].append(algo)
    elif not is_tau and not is_size and is_lr:
        algos_grp['SAC_lr'].append(algo)
    elif is_tau or is_size or is_lr:
        algos_grp['SAC_combo'].append(algo)


algos_grp['TD3_tau'] = ['TD3-AmelPredSto']
algos_grp['TD3_size'] = ['TD3-AmelPredSto']
algos_grp['TD3_lr'] = ['TD3-AmelPredSto']
algos_grp['TD3_combo'] = ['TD3-AmelPredSto']

for algo in algos_grp['TD3']:
    is_tau = '_tau_' in algo
    is_size = '_size_' in algo
    is_lr = '_lr_' in algo

    if is_tau and not is_size and not is_lr:
        algos_grp['TD3_tau'].append(algo)
    elif not is_tau and is_size and not is_lr:
        algos_grp['TD3_size'].append(algo)
    elif not is_tau and not is_size and is_lr:
        algos_grp['TD3_lr'].append(algo)
    elif is_tau or is_size or is_lr:
        algos_grp['TD3_combo'].append(algo)

algos_grp['comparison'] = ['TD3-Ni'] + algos_grp['SAC'] + algos_grp['TD3']
algos_grp['comparison'] = ['TD3-Ni', 'SAC-AmelPredSto_best', 'TD3-AmelPredSto_best']

# %% Aggregated reward
print('Plotting aggregated reward')
algos = list(rewards_eval.keys())
algos.sort()

# alg_norm = 'TD3-Ni'
# alg_norm = 'TD3-AmelPredDet'
# alg_norm = 'TD3-AmelPred-APESto'


for grp_key, grp in algos_grp.items():
    # if grp_key != 'SAC':
    #     continue
    print('Processing', grp_key, 'norm alg.', alg_norm)
    metric2plot = ['Median', 'IQM', 'Mean', 'Optimality Gap']
    fig, axes = plot_aggregated_metrics(grp, copy.deepcopy(rewards_eval), alg_norm, metric2plot)

    if out_path is not None:
        fig.savefig(out_path / f"rliable_aggregated_metrics_{grp_key}.pdf", bbox_inches='tight', pad_inches=0.1)
    fig.show()

# %% Sample efficiency curve
print('Plotting Sample efficiency curve')

for grp_key, grp in algos_grp.items():
    # if grp_key in ['SAC', 'TD3']:
    #     continue
    print('Processing', grp_key, 'norm alg.', alg_norm)
    fig, ax = plot_sample_efficiency(grp, copy.deepcopy(rewards_eval), alg_norm, np.array(range(rewards_eval[alg_norm].shape[-1])))
    if out_path is not None:
        fig.savefig(out_path / f"rliable_efficiency_curve_{grp_key}.pdf", bbox_inches='tight')
    fig.show()

# %% Performance profile
print('Plotting Performance profile')

for grp_key, grp in algos_grp.items():
    # if grp_key in ['SAC', 'TD3']:
    #     continue
    print('Processing', grp_key, 'norm alg.', alg_norm)
    fig, axes = plot_performance_profile(grp, copy.deepcopy(rewards_eval), alg_norm)
    if out_path is not None:
        fig.savefig(out_path / f"rliable_performance_profile_{grp_key}.pdf", bbox_inches='tight')
    fig.show()

# %% Probability of Improvement
print('Plotting Probability of Improvement')

# Target coordinates
algos_pairs = {
    'TD3': [
        ('TD3-Ni', 'TD3-AmelPredSto'),
        ('TD3-Ni', 'TD3-AmelPredSto*'),
        ('TD3-Ni', 'TD3-AmelPredSto**'),
        ('TD3-AmelPredSto**', 'TD3-Ni'),
        ('TD3-AmelPredSto**', 'TD3-AmelPredSto'),
        ('TD3-AmelPredSto**', 'TD3-AmelPredSto*'),
        ('TD3-AmelPredSto**', 'SAC-AmelPredSto**'),
        ],
    'SAC': [
        ('TD3-Ni', 'SAC-AmelPredSto'),
        ('TD3-Ni', 'SAC-AmelPredSto*'),
        ('TD3-Ni', 'SAC-AmelPredSto**'),
        ('SAC-AmelPredSto**', 'TD3-Ni'),
        ('SAC-AmelPredSto**', 'SAC-AmelPredSto'),
        ('SAC-AmelPredSto**', 'SAC-AmelPredSto*'),
        ('SAC-AmelPredSto**', 'TD3-AmelPredSto**'),
        ],
}

for pair_key, pair in algos_pairs.items():
    print('Processing', pair_key)
    fig, axes = plot_probability_improvement(algos, copy.deepcopy(rewards_eval), alg_norm, pair)
    if out_path is not None:
        fig.savefig(out_path / f"rliable_probability_improvement_{pair_key}.pdf", bbox_inches='tight')
    fig.show()



