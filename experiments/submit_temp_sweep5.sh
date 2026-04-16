#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=a6000ada
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --job-name=exp
#SBATCH --output=logs/slurm_sweep5_%j.out

set -o pipefail

source ~/.bashrc
conda activate pmgolf_stable
cd /data/AmitRoyChowdhury/vedant/parameter-golf

# Shared baseline config (exp_12 architecture)
export DATA_PATH="/data/AmitRoyChowdhury/vedant/parameter-golf/data/datasets/fineweb10B_sp1024/"
export TOKENIZER_PATH="/data/AmitRoyChowdhury/vedant/parameter-golf/data/tokenizers/fineweb_1024_bpe.model"
export VOCAB_SIZE=1024
export NUM_LAYERS=11
export MODEL_DIM=512
export MLP_MULT=2
export MATRIX_LR=0.04
export WARMDOWN_ITERS=3500
export DEPTH_RECURRENCE=2
export RECURRENCE_START_LAYER=4
export RECURRENCE_END_LAYER=5
export ITERATIONS=20000
export TRAIN_BATCH_TOKENS=131072
export TRAIN_SEQ_LEN=1024
export VAL_LOSS_EVERY=500
export TRAIN_LOG_EVERY=100
export MAX_WALLCLOCK_SECONDS=21600
export WARMUP_STEPS=5

run_case () {
  local run_id="$1"
  local qk="$2"
  local wd="$3"
  local ema_decay="$4"
  local ema_start="$5"

  echo "================================================"
  echo "Starting ${run_id} (QK=${qk}, WD=${wd}, EMA=${ema_decay}@${ema_start})"
  echo "================================================"

  QK_GAIN_INIT="$qk" \
  WEIGHT_DECAY="$wd" \
  EMA_DECAY="$ema_decay" \
  EMA_START_STEP="$ema_start" \
  RUN_ID="$run_id" \
  torchrun --standalone --nproc_per_node=1 /data/AmitRoyChowdhury/vedant/parameter-golf/train_gpt.py \
    2>&1 | tee "logs/${run_id}.txt"
}

# 4 tuning runs
run_case "exp12_qk3_wd0" "3.0" "0.0" "0.0" "0" || exit 1
run_case "exp12_qk25_wd0" "2.5" "0.0" "0.0" "0" || exit 1
run_case "exp12_qk3_wd0005" "3.0" "0.005" "0.0" "0" || exit 1
run_case "exp12_qk3_wd001" "3.0" "0.01" "0.0" "0" || exit 1

# EMA run on current concluded params
run_case "exp12_ema_qk5_wd0" "5.0" "0.0" "0.995" "1000" || exit 1

echo "All 5 runs completed."
