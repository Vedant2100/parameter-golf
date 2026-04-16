#!/usr/bin/env bash
set -euo pipefail

# RunPod one-file launcher for exp13-style run on 8xH100.
# Usage:
#   bash runpod_exp13.sh
# Optional overrides:
#   RUN_ID=exp13_try2 MAX_WALLCLOCK_SECONDS=600 bash runpod_exp13.sh
#   AUTO_SEND_BACK=1 bash runpod_exp13.sh
#   AUTO_SEND_BACK=1 SEND_INCLUDE_FULL_LOGS=0 bash runpod_exp13.sh

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DATA_PATH="${DATA_PATH:-$REPO_ROOT/data/datasets/fineweb10B_sp1024/}"
TOKENIZER_PATH="${TOKENIZER_PATH:-$REPO_ROOT/data/tokenizers/fineweb_1024_bpe.model}"
RUN_ID="${RUN_ID:-exp_13_runpod_8xh100}"
AUTO_SEND_BACK="${AUTO_SEND_BACK:-1}"
SEND_INCLUDE_FULL_LOGS="${SEND_INCLUDE_FULL_LOGS:-1}"

echo "Repo root: $REPO_ROOT"
echo "Run ID: $RUN_ID"
echo "Data path: $DATA_PATH"
echo "Tokenizer path: $TOKENIZER_PATH"

if [ ! -f "$TOKENIZER_PATH" ] || [ ! -d "$DATA_PATH" ]; then
  echo "Dataset/tokenizer missing. Downloading sp1024 artifacts..."
  python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 80
fi

if [ ! -f "$TOKENIZER_PATH" ]; then
  echo "Tokenizer still missing at: $TOKENIZER_PATH"
  exit 1
fi

if [ ! -d "$DATA_PATH" ]; then
  echo "Dataset dir still missing at: $DATA_PATH"
  exit 1
fi

echo "Starting torchrun on 8 GPUs..."
TORCH_LAUNCHER=()
if command -v torchrun >/dev/null 2>&1; then
  TORCH_LAUNCHER=(torchrun)
else
  if command -v python3 >/dev/null 2>&1; then
    TORCH_LAUNCHER=(python3 -m torch.distributed.run)
  elif command -v python >/dev/null 2>&1; then
    TORCH_LAUNCHER=(python -m torch.distributed.run)
  else
    echo "Neither torchrun nor python is available in PATH."
    exit 1
  fi
  echo "torchrun not found; using fallback launcher: ${TORCH_LAUNCHER[*]}"
fi

RUN_ID="$RUN_ID" \
DATA_PATH="$DATA_PATH" \
TOKENIZER_PATH="$TOKENIZER_PATH" \
VOCAB_SIZE="${VOCAB_SIZE:-1024}" \
NUM_LAYERS="${NUM_LAYERS:-11}" \
MODEL_DIM="${MODEL_DIM:-512}" \
MLP_MULT="${MLP_MULT:-2}" \
MATRIX_LR="${MATRIX_LR:-0.04}" \
WARMDOWN_ITERS="${WARMDOWN_ITERS:-3500}" \
DEPTH_RECURRENCE="${DEPTH_RECURRENCE:-2}" \
RECURRENCE_START_LAYER="${RECURRENCE_START_LAYER:-4}" \
RECURRENCE_END_LAYER="${RECURRENCE_END_LAYER:-5}" \
ITERATIONS="${ITERATIONS:-20000}" \
TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS:-524288}" \
TRAIN_SEQ_LEN="${TRAIN_SEQ_LEN:-1024}" \
VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-500}" \
TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-100}" \
MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-600}" \
WARMUP_STEPS="${WARMUP_STEPS:-5}" \
QK_GAIN_INIT="${QK_GAIN_INIT:-2.5}" \
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}" \
EMA_DECAY="${EMA_DECAY:-0.995}" \
EMA_START_STEP="${EMA_START_STEP:-1000}" \
"${TORCH_LAUNCHER[@]}" --standalone --nproc_per_node=8 train_gpt.py

echo "Done. Check logs/${RUN_ID}.txt for metrics."

# Optionally prepare and send outputs back to local machine.
OUTPUT_ARCHIVE="${RUN_ID}_outputs.tar.gz"
if [ "$SEND_INCLUDE_FULL_LOGS" = "1" ]; then
  tar -czf "$OUTPUT_ARCHIVE" logs "final_model.int8.ptz" "final_model.pt"
else
  # Smaller bundle: only this run's log + core artifacts.
  tar -czf "$OUTPUT_ARCHIVE" "logs/${RUN_ID}.txt" "final_model.int8.ptz" "final_model.pt"
fi
echo "Created output bundle: $OUTPUT_ARCHIVE"

if [ "$AUTO_SEND_BACK" = "1" ]; then
  if command -v runpodctl >/dev/null 2>&1; then
    echo "Sending bundle via runpodctl..."
    runpodctl send "$OUTPUT_ARCHIVE"
  else
    echo "runpodctl not found in PATH; cannot auto-send."
    echo "Manual send command:"
    echo "  runpodctl send $OUTPUT_ARCHIVE"
  fi
else
  echo "Auto send disabled."
  echo "To send results back manually:"
  echo "  runpodctl send $OUTPUT_ARCHIVE"
fi
