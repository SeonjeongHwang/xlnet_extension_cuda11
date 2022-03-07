#!/bin/bash -l

#SBATCH -J  coqa-base       # --job-name=singularity
#SBATCH -o  coqa-base.%j.out   # slurm output file (%j:expands to %jobId)
#SBATCH -p  A100-80GB             # queue or partiton name ; sinfo  output
#SBATCH --gres=gpu:1          # gpu Num Devices  가급적이면  1,2,4.6,8  2배수로 합니다.
GPU_NUM=4
TAG=base

echo $GPU_NUM

export SINGULARITYENV_CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES

module purge
module load singularity

OUTPUT_DIR=output/coqa/data
MODEL_DIR=output/coqa/$TAG

mkdir $OUTPUT_DIR
mkdir $MODEL_DIR
#singularity exec --nv ./xln_24_latest.sif python run_coqa.py \
#    --spiece-model-file=model/cased_large/spiece.model \
#    --model-config-path=model/cased_large/xlnet_config.json \
#    --init-checkpoint=model/cased_large/xlnet_model.ckpt \
#    --task-name=coqa \
#    --random-seed=2020 \
#    --predict-tag=$TAG \
#    --data-dir=data/coqa \
#    --output-dir=$OUTPUT_DIR \
#    --model-dir=$MODEL_DIR \
#    --export-dir=output/coqa/export \
#    --max-seq-length=512 \
#    --max-query-length=128 \
#    --train-batch-size=12 \
#    --num-hosts=1 \
#    --num-core-per-host=$GPU_NUM \
#    --learning-rate=3e-5 \
#    --train-steps=6000 \
#    --save-steps=3000 \
#    --do-train \
#    --do-predict
    
python tool/convert_coqa.py \
--input_file=$OUTPUT_DIR/predict.$TAG.summary.json \
--output_file=$OUTPUT_DIR/predict.$TAG.span.json

python tool/eval_coqa.py \
--data-file=data/coqa/dev-coqa.json \
--pred-file=$OUTPUT_DIR/predict.$TAG.span.json \
> $OUTPUT_DIR/predict.$TAG.eval.json