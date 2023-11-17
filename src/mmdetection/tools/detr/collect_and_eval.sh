START=$PWD

cd $WORK 

singularity run --nv -H $WORK --bind /work:/work $WORK/python.sif python \
    $WORK/mmdetection/tools/detr/collect_output.py \

singularity run --nv -H $WORK --bind /work:/work $WORK/python.sif python \
    $WORK/mmdetection/tools/detr/coco_eval_from_pkl.py \


cd $START
