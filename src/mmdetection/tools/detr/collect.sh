START=$PWD

cd $WORK 

singularity run --nv -H $WORK --bind /work:/work $WORK/python.sif python \
    $WORK/mmdetection/tools/detr/collect_output.py \
    $WORK/detr_r50_16x4_5e_output_heads_only/detr_r50_4x16_5e_output_heads_only_seed_0/detr_r50_4x16_5e_output_heads_only.py \


cd $START
