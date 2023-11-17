START=$PWD

cd $WORK 

#this should be able to be changed
LOC=src/mmdetection/exps/detr_r50_16x4_5e_output_heads_only

for dir in $LOC/detr*; do
    config=$dir/detr_r50_4x16_5e_output_heads_only.py 
    checkpoint=$dir/latest.pth 

    singularity run --nv -H $WORK --bind /work:/work sif/python.sif python \
        src/mmdetection/tools/detr/collect_output.py \
        $config \
        $checkpoint \
        $dir/output.pkl
    
    singularity run --nv -H $WORK --bind /work:/work sif/python.sif python \
        src/mmdetection/tools/detr/coco_eval_from_pkl.py \
        $dir/output.pkl \
        $dir/results.json
done

cd $START
