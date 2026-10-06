#!/bin/bash
CACHE=$1
python train.py task=SimpleHandHora headless=False pipeline=gpu \
task.env.numEnvs=1 test=True \
task.env.object.type=cube \
task.env.randomization.randomizeMass=False \
task.env.randomization.randomizeCOM=False \
task.env.randomization.randomizeFriction=False \
task.env.randomization.randomizePDGains=False \
task.env.randomization.randomizeScale=True \
train.algo=ProprioAdapt \
train.ppo.priv_info=True train.ppo.proprio_adapt=True \
train.ppo.output_name=Hand2Hora/"${CACHE}" \
checkpoint=outputs/Hand2Hora/"${CACHE}"/stage2_nn/model_last.ckpt
