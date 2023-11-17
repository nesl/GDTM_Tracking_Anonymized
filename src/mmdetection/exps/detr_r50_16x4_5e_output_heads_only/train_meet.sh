module load singularity
for seed in {100..1000..100}; do
    srun -p gypsum-titanx-phd -G 4 -c 24 --mem=150GB \
        singularity run --nv -H $WORK \
            python.sif bash \
            $WORK/mmdetection/tools/dist_train.sh \
            $WORK/mmdetection/configs/detr/detr_r50_4x16_5e_output_heads_only.py \
            4 \
            --seed $seed \
            --work-dir $WORK/logs/detr_r50_4x16_5e_output_heads_only_seed_$seed &
done
