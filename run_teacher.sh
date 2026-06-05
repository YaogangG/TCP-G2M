#!/usr/bin/env bash
set -e
DATASET=${1:-chameleon}
SETTING=${2:-tran}
DEVICE=${3:-cuda:0}
python train.py --dataset ${DATASET} --exp_setting ${SETTING} --teacher SAGE --student MLP --distill_mode 0 --device ${DEVICE}
