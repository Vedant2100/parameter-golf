#!/bin/bash
set -o pipefail

# --- Environment Setup ---
# This script is designed for RTX 6000 Ada and H100 GPUs.
# Ensure you have activated the correct environment: conda activate pmgolf_stable

# --- Default Configuration for "Production" Runs ---
# Each variable respects an env-var override so you can do:
#   TRAIN_BATCH_TOKENS=32768 MAX_WALLCLOCK_SECONDS=21600 ./run_experiments.sh 9b
DATA_PATH="${DATA_PATH:-/data/AmitRoyChowdhury/vedant/parameter-golf/data/datasets/fineweb10B_sp1024/}"
TOKENIZER_PATH="${TOKENIZER_PATH:-/data/AmitRoyChowdhury/vedant/parameter-golf/data/tokenizers/fineweb_1024_bpe.model}"
VOCAB_SIZE="${VOCAB_SIZE:-1024}"
ITERATIONS="${ITERATIONS:-20000}"
TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS:-131072}"
VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-500}"
TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-100}"
MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-600}"
WARMUP_STEPS="${WARMUP_STEPS:-5}"
TRAIN_SEQ_LEN="${TRAIN_SEQ_LEN:-1024}"
QK_GAIN_INIT="${QK_GAIN_INIT:-1.5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.04}"

# Ensure logs directory exists
mkdir -p logs

# --- Command Line Argument ---
# Usage: ./run_experiments.sh [exp_id]
# Example: ./run_experiments.sh 8
TARGET_EXP=$1

function run_exp() {
    local run_id=$1
    local exp_num=$2
    local num_layers=${3:-9}
    local model_dim=${4:-512}
    local mlp_mult=${5:-2}
    local matrix_lr=${6:-0.04}
    local warmdown_iters=${7:-1200}
    local depth_recurrence=${8:-1}
    local recurrence_start_layer=${9:--1}
    local recurrence_end_layer=${10:--1}
    
    # If a target was provided, skip if this doesn't match
    if [ -n "$TARGET_EXP" ] && [ "$TARGET_EXP" != "$exp_num" ] && [ "$TARGET_EXP" != "$run_id" ]; then
        return
    fi

    echo "------------------------------------------------"
    echo "🚀 Starting Experiment: $run_id"
    echo "   Layers: $num_layers | Dim: $model_dim | MLP: ${mlp_mult}x"
    echo "   Muon LR: $matrix_lr | Warmdown: $warmdown_iters"
    echo "   QK Gain: $QK_GAIN_INIT | Weight Decay: $WEIGHT_DECAY"
    echo "   Recurrence: depth=$depth_recurrence window=${recurrence_start_layer}-${recurrence_end_layer}"
    echo "   Batch Size: $TRAIN_BATCH_TOKENS | Max Steps: $ITERATIONS"
    echo "------------------------------------------------"

    NUM_LAYERS=$num_layers \
    MODEL_DIM=$model_dim \
    MLP_MULT=$mlp_mult \
    MATRIX_LR=$matrix_lr \
    WARMDOWN_ITERS=$warmdown_iters \
    DEPTH_RECURRENCE=$depth_recurrence \
    RECURRENCE_START_LAYER=$recurrence_start_layer \
    RECURRENCE_END_LAYER=$recurrence_end_layer \
    ITERATIONS=$ITERATIONS \
    TRAIN_BATCH_TOKENS=$TRAIN_BATCH_TOKENS \
    TRAIN_SEQ_LEN=$TRAIN_SEQ_LEN \
    VAL_LOSS_EVERY=$VAL_LOSS_EVERY \
    TRAIN_LOG_EVERY=$TRAIN_LOG_EVERY \
    MAX_WALLCLOCK_SECONDS=$MAX_WALLCLOCK_SECONDS \
    WARMUP_STEPS=$WARMUP_STEPS \
    QK_GAIN_INIT=$QK_GAIN_INIT \
    WEIGHT_DECAY=$WEIGHT_DECAY \
    DATA_PATH=$DATA_PATH \
    TOKENIZER_PATH=$TOKENIZER_PATH \
    VOCAB_SIZE=$VOCAB_SIZE \
    RUN_ID=$run_id \
    torchrun --standalone --nproc_per_node=1 /data/AmitRoyChowdhury/vedant/parameter-golf/train_gpt.py 2>&1 | tee logs/${run_id}.txt
    local run_status=$?
    if [ $run_status -ne 0 ]; then
        echo "❌ Experiment $run_id failed (exit=$run_status). See logs/${run_id}.txt"
        return $run_status
    fi

    echo "✅ Experiment $run_id complete. Log saved to logs/${run_id}.txt"
}

# --- Experiment Suite ---

# 1. Baseline
run_exp "exp_01_baseline" "1" 9 512 2 0.04 1200 || exit $?

# 2. Increased Depth
run_exp "exp_02_11_layers" "2" 11 512 2 0.04 1200 || exit $?

# 3. Increased Width
run_exp "exp_03_width_576" "3" 9 576 2 0.04 1200 || exit $?

# 4. Bigger MLP
run_exp "exp_04_3x_mlp" "4" 9 512 3 0.04 1200 || exit $?

# 5. Higher Muon LR
run_exp "exp_05_lr_05" "5" 9 512 2 0.05 1200 || exit $?

# 6. Longer Warmdown
run_exp "exp_06_warmdown_3500" "6" 9 512 2 0.04 3500 || exit $?

# 7. Stacked Combo
run_exp "exp_07_stacked_combo" "7" 11 512 3 0.05 3500 || exit $?

# 8. Wide & Fat (Large Batch)
run_exp "exp_08_wide_and_fat" "8" 9 512 3.5 0.04 3500 || exit $?

# ----------------------------------------------------
# 9. THE FINAL SPRINT: High-Frequency Wide & Fat
# ----------------------------------------------------
# Configuration: 
# - Batch Size: 131,072 (4x faster updates)
# - 9 Layers, 512 Dim, 3.5x MLP Multiplier
# ----------------------------------------------------
run_exp "exp_09_high_freq_final" "9" 9 512 3.5 0.04 3500 || exit $?


# ----------------------------------------------------
# 10. FAIR BASELINE: High-Frequency Standard (2.0x MLP)
# ----------------------------------------------------
run_exp "baseline" "baseline" 9 512 2 0.04 3500 || exit $?

# Recurrence experiments derived from exp_09
# 9a: Depth recurrence with 3 unique repeats (DEPTH_RECURRENCE=3)
run_exp "exp_09_recur3" "9a" 9 512 3.5 0.04 3500 3 || exit $?

# 9b: Depth recurrence with 2 over layers 4-5 (leaderboard-inspired loop style)
run_exp "exp_09_recur2_loop45" "9b" 9 512 3.5 0.04 3500 2 4 5 || exit $?

# ----------------------------------------------------
# 10. Leaderboard-inspired: 11 layers, 4x MLP, high WD, loop 4-5 x2
# Based on leader entries #5-#8 (1.085-1.098 BPB on 8xH100)
# ----------------------------------------------------
run_exp "exp_10_leader_inspired" "10" 11 512 4 0.04 3500 2 4 5 || exit $?

# ----------------------------------------------------
# 11. Size-safe baseline improver: 11 effective layers via recurrence, 2x MLP
# Intended to stay closer to 16MB while keeping recurrence + WD gains.
# ----------------------------------------------------
run_exp "exp_11_fit16mb" "11" 11 512 2 0.04 3500 2 4 5 || exit $?

# ----------------------------------------------------
# 12. Same as exp_11 but intended for WEIGHT_DECAY=0.0 override at launch.
# ----------------------------------------------------
run_exp "exp_12_fit16mb_wd0" "12" 11 512 2 0.04 3500 2 4 5 || exit $?

# Quick smoke target (only runs when TARGET_EXP=smoke)
if [ "$TARGET_EXP" = "smoke" ]; then
    echo "🔬 Running smoke recurrence test (smoke_recur3)..."
    run_exp "smoke_recur3" "smoke" 9 512 3.5 0.04 10 3 4 5 || exit $?
    echo "✅ Smoke run complete. Log: logs/smoke_recur3.txt"
    exit 0
fi


if [ -z "$TARGET_EXP" ]; then
    echo "🎉 ALL EXPERIMENTS COMPLETE!"
else
    echo "🏁 Targeted run '$TARGET_EXP' finished."
fi

# --- Cluster launch cheat-sheet ---
# Best GPU:     srun -p gpu    --gres=gpu:1 --constraint=a6000ada --cpus-per-task=8 --mem=32G --time=08:00:00 --pty bash -l
# Backup GPU:   srun -p vcggpu --gres=gpu:1 --constraint=rtx3090  --cpus-per-task=8 --mem=32G --time=08:00:00 --pty bash -l
#
# Then:
#   conda activate pmgolf_stable
#   cd /data/AmitRoyChowdhury/vedant/parameter-golf
#
# Equivalent of 10 min on 8xH100 (~6h on a6000ada, ~8h on rtx3090):
#   MAX_WALLCLOCK_SECONDS=21600 QK_GAIN_INIT=5.0 ./run_experiments.sh 10