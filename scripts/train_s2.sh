#!/bin/bash
GPUS=$1
SEED=$2
CACHE=$3

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:3:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

echo extra "${EXTRA_ARGS}"

CUDA_VISIBLE_DEVICES=${GPUS} \
python train.py task=SimpleHandHora headless=True seed=${SEED} \
task.env.numEnvs=20000 \
task.env.object.type=cube \
task.env.forceScale=2 task.env.randomForceProbScalar=0.25 \
train.algo=ProprioAdapt \
train.ppo.priv_info=True train.ppo.proprio_adapt=True \
train.ppo.output_name=Hand2Hora/"${CACHE}" \
checkpoint=outputs/Hand2Hora/"${CACHE}"/stage1_nn/best.pth \
${EXTRA_ARGS}
