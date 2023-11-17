module load singularity

#srun -p gypsum-1080ti-phd -G 8 -c 24 --mem=150GB --exclude=ials-gpu015,gpu003,gpu004,ials-gpu006,ials-gpu033,ials-gpu029 \
START=$PWD
cd $WORK
for seed in {400000..700000..100000}; do
    srun -p gypsum-1080ti-phd -G 8 -c 24 --mem=150GB \
        singularity run --nv -H $WORK --bind /work:/work \
            sif/python.sif bash \
            src/mmdetection/tools/dist_train.sh \
            src/mmdetection/configs/detr/detr_r50_8x16_150e_decoder_and_output.py \
            8 \
            --seed $seed \
            --work-dir logs/detr_r50_8x16_150e_decoder_and_output_$seed &
done
cd $START
