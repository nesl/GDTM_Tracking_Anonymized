
START=$PWD
cd $WORK

singularity run --nv -H $WORK sif/python.sif bash \
    $WORK/mmdetection/tools/dist_train.sh \
    $WORK/mmdetection/configs/detr/detr_r50_8x2_200e_coco.py \
    8 \
    --seed 100 \
    --work-dir $WORK/mmdetection/exps/detr_r50_8x2_200e_coco

cd $START
