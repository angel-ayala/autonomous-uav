# Code for State Representation Learning baseline experiments

This code was used to train RL agents with Ni et al. proposed baselines methods.

## Usage examples

Minimalist $\phi_L$ with L2 objective and EMA targets:
```bash
python train.py algo=ours aux=l2 aux_optim=ema aux_coef=v-10.0
```

Minimalist $\phi_L$ with L2 objective and detached targets:
```bash
python train.py algo=ours aux=l2 aux_optim=detach aux_coef=v-10.0
```

Minimalist $\phi_L$ with L2 objective and online targets:
```bash
python train.py algo=ours aux=l2 aux_optim=online aux_coef=v-10.0
```

Minimalist $\phi_L$ with forward KL objective and EMA targets:
```bash
python train.py algo=ours aux=fkl aux_optim=ema aux_coef=v-10.0
```

Minimalist $\phi_L$ with forward KL objective and detached targets:
```bash
python train.py algo=ours aux=fkl aux_optim=detach aux_coef=v-10.0
```

Minimalist $\phi_L$ with forward KL objective and online targets:
```bash
python train.py algo=ours aux=fkl aux_optim=online aux_coef=v-10.0
```

Minimalist $\phi_L$ with reverse KL objective and EMA targets:
```bash
python train.py algo=ours aux=rkl aux_optim=ema aux_coef=v-1.0
```

Minimalist $\phi_L$ with reverse KL objective and detached targets:
```bash
python train.py algo=ours aux=rkl aux_optim=detach aux_coef=v-1.0
```

Minimalist $\phi_L$ with reverse KL objective and online targets:
```bash
python train.py algo=ours aux=rkl aux_optim=online aux_coef=v-1.0
```

You will see the logging and executed config files in `logs/` folder.

## Acknowledgement
This code derives from [Bridging State and History Representations: Understanding Self-Predictive RL](https://arxiv.org/abs/2401.08898), authored by *Tianwei Ni, Benjamin Eysenbach, Erfan Seyedsalehi, Michel Ma, Clement Gehring, Aditya Mahajan, and Pierre-Luc Bacon.*

The repository provides a code adaptation of the Mujoco code for the [Webots Drone Scene](https://github.com/angel-ayala/gym-webots-drone).
For more details please visit the [original code repository](https://github.com/twni2016/self-predictive-rl/tree/main/mujoco_code)
