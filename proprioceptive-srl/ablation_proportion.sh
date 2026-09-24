#!/bin/bash
# Proportion between feature and latent dimension size comparison

# environment and seeds
ENV_NAME=Ant-v4
SEEDS=(202501 202502 202503 202504 202505)
# algorithm
SCRIPT=learn_td3_mujoco.py
# deterministic model
REP_MODEL=(--is-srl --model-ispr)
# stochastic model
#REP_MODEL=(--is-srl --model-ispr --use-stochastic)
# feature-latent proportionss
FEAT_LAT_PROPS=("1:1" "1:0.5" "0.5:1" "1:2" "2:1")

for SEED in "${SEEDS[@]}"; do
  for PROP in "${FEAT_LAT_PROPS[@]}"; do
    echo "Running $SCRIPT: $ENV_NAME --seed $SEED ${REP_MODEL[@]} --feat-lat-prop $PROP"
    python $SCRIPT --environment-id $ENV_NAME --seed $SEED --use-cuda "${REP_MODEL[@]}" --joint-optimization --feat-lat-prop "$PROP"
    sleep 2

  done
done

