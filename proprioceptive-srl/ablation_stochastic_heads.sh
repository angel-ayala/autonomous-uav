#!/bin/bash
# Regularization methods comparison

ENV_NAME=Ant-v4
SEEDS=(202501 202502 202503 202504 202505)
# algorithm
SCRIPT=learn_td3_mujoco.py
# stochastic model
REP_MODEL=(--is-srl --joint-optimization --model-ispr --use-stochastic)
# feature-latent proportionss
DIST_HEADS=(--dist-bound --dist-bound-norm --dist-bound-logvar --dist-bound-logvar-norm)

for SEED in "${SEEDS[@]}"; do
  for DHEAD in "${DIST_HEADS[@]}"; do
    echo "Running $SCRIPT: $ENV_NAME --seed $SEED ${REP_MODEL[@]} $DHEAD"
    python $SCRIPT --environment-id $ENV_NAME --seed $SEED --use-cuda "${REP_MODEL[@]}" $DHEAD
    sleep 2

  done
done

