#!/bin/bash
# Vanilla algorithms and proposed variants comparison

# environment and seeds
ENV_NAME=Humanoid-v4
SEEDS=(202601 202602 202603 202604 202605)
# algorithm
SCRIPT=learn_td3_mujoco.py
#SCRIPT=learn_sac_mujoco.py
# variants
SRL_MODEL=(--is-srl --joint-optimization --use-mamba-dec)
SRL_STO_MODEL=("${SRL_MODEL[@]}" --use-stochastic)
# models
MODELS=(--model-ispr --model-proprio)

for SEED in "${SEEDS[@]}"; do
  for REP_MODEL in "${MODELS[@]}"; do
    # Deterministic
    echo "Running $ALGO: $ENV_NAME --seed $SEED $REP_MODEL ${SRL_MODEL[@]}"
    python "$SCRIPT" --environment-id "$ENV_NAME" --seed "$SEED" --use-cuda "${SRL_MODEL[@]}" $REP_MODEL
    sleep 2
  
    # Stochastic
    echo "Running $ALGO: $ENV_NAME --seed $SEED $REP_MODEL ${SRL_STO_MODEL[@]}"
    python "$SCRIPT" --environment-id "$ENV_NAME" --seed "$SEED" --use-cuda "${SRL_STO_MODEL[@]}" $REP_MODEL
    sleep 2

  done
done

