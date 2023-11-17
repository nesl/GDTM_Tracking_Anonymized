#!/usr/bin/env bash
CONFIG=$1
NAME=$(basename $1 .py)
EXPDIR=logs/$NAME

mkdir -p $EXPDIR/val

python tools/cache_datasets.py $CONFIG
tools/dist_test.sh $CONFIG $2 --checkpoint $EXPDIR/latest.pth --eval track vid --cfg-options evaluation.logdir=$EXPDIR/val evaluation.video_length=5 evaluation.dataset=val evaluation.grid_search=True
