#!/bin/bash
# CACHE can be some existing output folder, does not matter
# numEnvs=20000, headless=True, episodeLen=50 to save time
# Generate grasp poses for the same object used by the cube rotation task.
# pipeline need to be cpu to get the pairwise contact
# no custom PD because bug in CPU mode
# mass should be about 50g
GPUS=$1
SCALE=$2
CUDA_VISIBLE_DEVICES=${GPUS} \
python gen_grasp.py task=SimpleHandGrasp headless=True pipeline=cpu \
task.env.numEnvs=20000 test=True \
task.env.controller.controlFrequencyInv=8 task.env.episodeLength=50 \
task.env.controller.torque_control=False task.env.genGrasps=True task.env.baseObjScale="${SCALE}" \
task.env.object.type=cube66 \
task.env.tactile.backend=contact_force \
task.env.randomization.randomizeMass=True task.env.randomization.randomizeMassLower=0.03 task.env.randomization.randomizeMassUpper=0.06 \
task.env.randomization.randomizeCOM=False \
task.env.randomization.randomizeFriction=False \
task.env.randomization.randomizePDGains=False \
task.env.randomization.randomizeScale=False \
train.ppo.priv_info=True
