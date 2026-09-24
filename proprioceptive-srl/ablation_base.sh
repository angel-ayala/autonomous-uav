#!/bin/bash
# Base algorithms comparison

# environment and seeds
ENV_NAME=Humanoid-v4
SEEDS=(202601 202602 202603 202604 202605)
# deterministic model
REP_MODEL=(--is-srl --model-ispr)
# stochastic model
#REP_MODEL=(--is-srl --model-ispr --use-stochastic)

for SEED in "${SEEDS[@]}"; do
  # TD3 AmelPred base feature 32
  echo "Running learn_td3_mujoco.py: $ENV_NAME --seed $SEED ${REP_MODEL[@]}"
  python learn_td3_mujoco.py --environment-id $ENV_NAME --seed $SEED --use-cuda "${REP_MODEL[@]}" --joint-optimization
  sleep 2
  
  # SAC AmelPred base feature 32
  echo "Running learn_sac_mujoco.py: $ENV_NAME --seed $SEED ${REP_MODEL[@]}"
  python learn_sac_mujoco.py --environment-id $ENV_NAME --seed $SEED --use-cuda "${REP_MODEL[@]}" --joint-optimization
  sleep 2

done

