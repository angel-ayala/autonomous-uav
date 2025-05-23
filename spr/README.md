# Code for Self-Predictive Representation baseline experiments

This code was used to train RL agents with Schwarzer et al. proposed method.

## Usage example:
```bash
python -m scripts.run_cf --public \
--frame-stack 3 \
--is-pixels \
--seed 202401
```

## What does each file do? 

    .
    ├── scripts
    │   └── run_cf.py                # The main training script to launch jobs.
    │   └── run_cf_eval.py           # The main evaluation script to launch jobs.
    ├── src                     
    │   ├── agent.py              # Implements the Agent API for action selection.
    │   ├── algos.py              # Distributional RL loss.
    │   ├── models.py             # Network architecture and forward passes.
    │   ├── rlpyt_drone_env.py    # Utility methods for env instanciation for rlpyt.
    │   ├── rlpyt_utils.py        # Utility methods that we use to extend rlpyt's functionality.
    │   └── utils.py              # Command line arguments and helper functions.
    │
    └── requirements.txt          # Dependencies

## Acknowledgement
This code derives from [Data-Efficient Reinforcement Learning with Self-Predictive Representations](https://arxiv.org/abs/2007.05929), authored by
*Max Schwarzer\*, Ankesh Anand\*, Rishab Goel, R Devon Hjelm, Aaron Courville, and Philip Bachman*.

The repository provides a code adaptation for the [Webots Drone Scene](https://github.com/angel-ayala/gym-webots-drone).
For more details please visit the [original code repository](https://github.com/mila-iqia/spr)
