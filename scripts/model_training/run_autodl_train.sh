#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-smoke}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

export HF_HOME="${HF_HOME:-/root/autodl-tmp/hf_cache}"
export NEMO_CACHE_DIR="${NEMO_CACHE_DIR:-/root/autodl-tmp/nemo_cache}"
mkdir -p "$HF_HOME" "$NEMO_CACHE_DIR"

ensure_mono_dir() {
  local input_dir="$1"
  local output_dir="$2"
  local input_count
  local output_count
  input_count="$(find "$input_dir" -maxdepth 1 -name '*.flac' | wc -l)"
  output_count="0"
  if [ -d "$output_dir" ]; then
    output_count="$(find "$output_dir" -maxdepth 1 -name '*.flac' | wc -l)"
  fi

  if [ "$input_count" != "$output_count" ]; then
    echo "Converting mono audio: $input_dir -> $output_dir ($output_count/$input_count ready)"
    python scripts/convert_to_mono.py \
      --input-dir "$input_dir" \
      --output-dir "$output_dir"
  else
    echo "Mono audio complete: $output_dir ($output_count/$input_count)"
  fi
}

TRAIN_MANIFEST="data/manifests/model_training/aishell4_train_L_sortformer_train.json"
VAL_MANIFEST="data/manifests/model_training/aishell4_train_L_sortformer_dev.json"
OUTPUT_DIR="results/model_training/sortformer_8spk"
MAX_EPOCHS=3
ACCUMULATE_GRAD_BATCHES=4
LR=1e-4
WEIGHT_DECAY=1e-3
POSITIVE_CLASS_WEIGHT=1.0
FREEZE_ENCODER_FLAG="--freeze-encoder"

case "$MODE" in
  smoke)
    MAX_EPOCHS=1
    ;;
  full)
    MAX_EPOCHS=3
    ;;
  long)
    MAX_EPOCHS=8
    ;;
  dense)
    echo "[dense] Rebuilding denser train/dev manifests: chunk=120s hop=30s"
    python scripts/model_training/prepare_sortformer_aishell4_train.py \
      --audio-dir train_L/train_L/wav_mono \
      --rttm-dir train_L/train_L/TextGrid \
      --output-dir data/manifests/model_training \
      --prefix aishell4_train_L_sortformer_dense \
      --max-sessions 30 \
      --dev-sessions 5 \
      --chunk-sec 120 \
      --chunk-hop-sec 30
    TRAIN_MANIFEST="data/manifests/model_training/aishell4_train_L_sortformer_dense_train.json"
    VAL_MANIFEST="data/manifests/model_training/aishell4_train_L_sortformer_dense_dev.json"
    OUTPUT_DIR="results/model_training/sortformer_8spk_dense"
    MAX_EPOCHS=16
    ACCUMULATE_GRAD_BATCHES=4
    LR=1e-4
    POSITIVE_CLASS_WEIGHT=3.0
    ;;
  dense_unfreeze)
    echo "[dense_unfreeze] Rebuilding denser train/dev manifests: chunk=120s hop=30s"
    python scripts/model_training/prepare_sortformer_aishell4_train.py \
      --audio-dir train_L/train_L/wav_mono \
      --rttm-dir train_L/train_L/TextGrid \
      --output-dir data/manifests/model_training \
      --prefix aishell4_train_L_sortformer_dense \
      --max-sessions 30 \
      --dev-sessions 5 \
      --chunk-sec 120 \
      --chunk-hop-sec 30
    TRAIN_MANIFEST="data/manifests/model_training/aishell4_train_L_sortformer_dense_train.json"
    VAL_MANIFEST="data/manifests/model_training/aishell4_train_L_sortformer_dense_dev.json"
    OUTPUT_DIR="results/model_training/sortformer_8spk_dense_unfreeze"
    MAX_EPOCHS=8
    ACCUMULATE_GRAD_BATCHES=8
    LR=3e-5
    POSITIVE_CLASS_WEIGHT=3.0
    FREEZE_ENCODER_FLAG="--no-freeze-encoder"
    ;;
  heavy)
    echo "[heavy] Rebuilding dense train/dev manifests: chunk=120s hop=15s"
    python scripts/model_training/prepare_sortformer_aishell4_train.py \
      --audio-dir train_L/train_L/wav_mono \
      --rttm-dir train_L/train_L/TextGrid \
      --output-dir data/manifests/model_training \
      --prefix aishell4_train_L_sortformer_heavy \
      --max-sessions 30 \
      --dev-sessions 5 \
      --chunk-sec 120 \
      --chunk-hop-sec 15
    TRAIN_MANIFEST="data/manifests/model_training/aishell4_train_L_sortformer_heavy_train.json"
    VAL_MANIFEST="data/manifests/model_training/aishell4_train_L_sortformer_heavy_dev.json"
    OUTPUT_DIR="results/model_training/sortformer_8spk_heavy"
    MAX_EPOCHS=20
    ACCUMULATE_GRAD_BATCHES=4
    LR=1e-4
    POSITIVE_CLASS_WEIGHT=3.0
    ;;
  lm_dense)
    ensure_mono_dir train_L/train_L/wav train_L/train_L/wav_mono
    ensure_mono_dir train_M/wav train_M/wav_mono
    echo "[lm_dense] Rebuilding train_L+train_M manifests: all 138 sessions, chunk=120s hop=30s"
    python scripts/model_training/prepare_sortformer_aishell4_train.py \
      --audio-dir train_L/train_L/wav_mono train_M/wav_mono \
      --rttm-dir train_L/train_L/TextGrid train_M/TextGrid \
      --output-dir data/manifests/model_training \
      --prefix aishell4_train_LM_sortformer_dense \
      --max-sessions 138 \
      --dev-sessions 12 \
      --min-speakers 3 \
      --max-speakers 8 \
      --chunk-sec 120 \
      --chunk-hop-sec 30
    TRAIN_MANIFEST="data/manifests/model_training/aishell4_train_LM_sortformer_dense_train.json"
    VAL_MANIFEST="data/manifests/model_training/aishell4_train_LM_sortformer_dense_dev.json"
    OUTPUT_DIR="results/model_training/sortformer_8spk_lm_dense"
    MAX_EPOCHS=16
    ACCUMULATE_GRAD_BATCHES=4
    LR=1e-4
    POSITIVE_CLASS_WEIGHT=3.0
    ;;
  lm5_dense)
    ensure_mono_dir train_L/train_L/wav train_L/train_L/wav_mono
    ensure_mono_dir train_M/wav train_M/wav_mono
    echo "[lm5_dense] Rebuilding train_L+train_M manifests: 5+ speakers only, chunk=120s hop=30s"
    python scripts/model_training/prepare_sortformer_aishell4_train.py \
      --audio-dir train_L/train_L/wav_mono train_M/wav_mono \
      --rttm-dir train_L/train_L/TextGrid train_M/TextGrid \
      --output-dir data/manifests/model_training \
      --prefix aishell4_train_LM_5plus_sortformer_dense \
      --max-sessions 90 \
      --dev-sessions 10 \
      --min-speakers 5 \
      --max-speakers 8 \
      --chunk-sec 120 \
      --chunk-hop-sec 30
    TRAIN_MANIFEST="data/manifests/model_training/aishell4_train_LM_5plus_sortformer_dense_train.json"
    VAL_MANIFEST="data/manifests/model_training/aishell4_train_LM_5plus_sortformer_dense_dev.json"
    OUTPUT_DIR="results/model_training/sortformer_8spk_lm5_dense"
    MAX_EPOCHS=16
    ACCUMULATE_GRAD_BATCHES=4
    LR=1e-4
    POSITIVE_CLASS_WEIGHT=3.0
    ;;
  *)
    echo "Usage: bash scripts/model_training/run_autodl_train.sh [smoke|full|long|dense|dense_unfreeze|heavy|lm_dense|lm5_dense]"
    exit 2
    ;;
esac

echo "Mode: $MODE"
echo "Train manifest: $TRAIN_MANIFEST"
echo "Val manifest: $VAL_MANIFEST"
echo "Output dir: $OUTPUT_DIR"
echo "Epochs: $MAX_EPOCHS | lr: $LR | pos_weight: $POSITIVE_CLASS_WEIGHT | accum: $ACCUMULATE_GRAD_BATCHES"

python scripts/model_training/train_sortformer_8spk.py \
  --model-path models/diar_sortformer_4spk-v1.nemo \
  --train-manifest "$TRAIN_MANIFEST" \
  --val-manifest "$VAL_MANIFEST" \
  --output-dir "$OUTPUT_DIR" \
  --max-speakers 8 \
  --batch-size 1 \
  --num-workers 2 \
  --accumulate-grad-batches "$ACCUMULATE_GRAD_BATCHES" \
  --max-epochs "$MAX_EPOCHS" \
  --lr "$LR" \
  --weight-decay "$WEIGHT_DECAY" \
  --positive-class-weight "$POSITIVE_CLASS_WEIGHT" \
  --precision bf16-mixed \
  --accelerator gpu \
  --devices 1 \
  $FREEZE_ENCODER_FLAG
