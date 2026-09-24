#!/bin/bash
# Regularization methods comparison

ENV_NAME=Ant-v4
SEEDS=(202501 202502 202503 202504 202505)
# algorithm
SCRIPT=learn_td3_mujoco.py
# deterministic model
#REP_MODEL=(--is-srl --model-ispr)
# stochastic model
#REP_MODEL=(--is-srl --model-ispr --use-stochastic)
# feature-latent proportionss
REGULARIZERS=("--enc-max-gradn 100" "--loss-balancer" "--enc-max-gradn 100 --loss-balancer ")

for SEED in "${SEEDS[@]}"; do
  for REG in "${REGULARIZERS[@]}"; do
    echo "Running $SCRIPT: $ENV_NAME --seed $SEED ${REP_MODEL[@]} $REG"
    python $SCRIPT --environment-id $ENV_NAME --seed $SEED --use-cuda ${REP_MODEL[@]} --joint-optimization $REG
    sleep 2

  done
done

