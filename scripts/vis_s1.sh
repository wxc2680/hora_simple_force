#!/bin/bash
CACHE=$1
python train.py task=SimpleHandHora headless=False pipeline=gpu \
task.env.numEnvs=1 test=True \
task.env.object.type=ball \
train.algo=PPO \
task.env.randomization.randomizeMass=False \
task.env.randomization.randomizeCOM=False \
task.env.randomization.randomizeFriction=False \
task.env.randomization.randomizePDGains=False \
task.env.randomization.randomizeScale=True \
task.env.randomization.randomizeScaleList='[1.0]' \
train.ppo.priv_info=True \
train.ppo.output_name=Hand2Hora/"${CACHE}" \
checkpoint=outputs/Hand2Hora/"${CACHE}"/stage1_nn/best.pth
