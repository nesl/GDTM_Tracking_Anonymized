#!/usr/bin/env bash

CONFIG=$1
NAME=$(basename $1 .py)
EXPDIR=logs/$NAME
# if [ -d $EXPDIR ]; then
#     rm -rf $EXPDIR/*
# fi
python tools/cache_datasets.py $CONFIG
tools/dist_train.sh $CONFIG $2 --work-dir $EXPDIR --seed 5 --resume-from $EXPDIR/latest.pth
