module load singularity

for seed in `seq 0 9`; do
    srun -p gypsum-titanx-phd -G 4 -c 24 --mem=150GB \
        singularity run --nv -H $WORK \
            sif/python.sif bash \
            src/mmdetection/tools/dist_train.sh \
            src/mmdetection/configs/detr/detr_r50_4x16_5e_output_heads_only.py \
            4 \
            --seed $seed \
            --work-dir logs/detr_r50_4x16_5e_output_heads_only_seed_$seed &
done
