#!/bin/bash
# Regularization methods comparison

# environment and seeds
ENV_NAME=Ant-v4
SEEDS=(202501 202502 202503 202504 202505)
# algorithm
SCRIPT=learn_td3_mujoco.py
#SCRIPT=learn_sac_mujoco.py
# variants
SRL_MODEL=(--is-srl --joint-optimization --model-ispr)
SRL_STO_MODEL=("${SRL_MODEL[@]}" --use-stochastic)
# regularization methods
REGULARIZERS=("--enc-max-gradn 100" "--loss-balancer" "--enc-max-gradn 100 --loss-balancer")

for SEED in "${SEEDS[@]}"; do
  for REG in "${REGULARIZERS[@]}"; do
    echo "Running $SCRIPT: $ENV_NAME --seed $SEED ${SRL_MODEL[@]} $REG"
    python $SCRIPT --environment-id $ENV_NAME --seed $SEED --use-cuda "${SRL_MODEL[@]}" $REG
    sleep 2

    echo "Running $SCRIPT: $ENV_NAME --seed $SEED ${SRL_STO_MODEL[@]} $REG"
    python $SCRIPT --environment-id $ENV_NAME --seed $SEED --use-cuda "${SRL_STO_MODEL[@]}" $REG
    sleep 2

  done
done

