#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=a6000ada
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --job-name=exp
#SBATCH --output=logs/slurm_exp13_%j.out

set -o pipefail

source ~/.bashrc
conda activate pmgolf_stable
cd /data/AmitRoyChowdhury/vedant/parameter-golf

# Baseline-beat target run:
# - exp12 architecture (11L, 512d, 2x MLP, recur 4-5 x2)
# - EMA enabled
# - QK tuned to 2.5
# - WD off (prior sweeps showed >0 hurts badly)
# - 4x token budget vs prior 131072 runs
DATA_PATH="/data/AmitRoyChowdhury/vedant/parameter-golf/data/datasets/fineweb10B_sp1024/" \
TOKENIZER_PATH="/data/AmitRoyChowdhury/vedant/parameter-golf/data/tokenizers/fineweb_1024_bpe.model" \
VOCAB_SIZE=1024 \
NUM_LAYERS=11 \
MODEL_DIM=512 \
MLP_MULT=2 \
MATRIX_LR=0.04 \
WARMDOWN_ITERS=3500 \
DEPTH_RECURRENCE=2 \
RECURRENCE_START_LAYER=4 \
RECURRENCE_END_LAYER=5 \
ITERATIONS=20000 \
TRAIN_BATCH_TOKENS=524288 \
TRAIN_SEQ_LEN=1024 \
VAL_LOSS_EVERY=500 \
TRAIN_LOG_EVERY=100 \
MAX_WALLCLOCK_SECONDS=25200 \
WARMUP_STEPS=5 \
QK_GAIN_INIT=2.5 \
WEIGHT_DECAY=0.0 \
EMA_DECAY=0.995 \
EMA_START_STEP=1000 \
RUN_ID=exp_13_baselinebeat_qk25_ema_tokens4x \
torchrun --standalone --nproc_per_node=1 /data/AmitRoyChowdhury/vedant/parameter-golf/train_gpt.py 2>&1 | tee logs/exp_13_baselinebeat_qk25_ema_tokens4x.txt
