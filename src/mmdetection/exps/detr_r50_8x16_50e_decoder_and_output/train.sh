module load singularity

START=$PWD
cd $WORK
for seed in `seq 0 9`; do
    srun -p gpu-long -G 8 -c 24 --mem=150GB --exclude=ials-gpu015 \
        singularity run --nv -H $WORK --bind /work:/work \
            sif/python.sif bash \
            src/mmdetection/tools/dist_train.sh \
            src/mmdetection/configs/detr/detr_r50_8x16_50e_decoder_and_output.py \
            8 \
            --seed $seed \
            --work-dir logs/detr_r50_8x16_50e_decoder_and_output_$seed &
done
cd $START
