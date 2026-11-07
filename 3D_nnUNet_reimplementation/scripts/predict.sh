#!/usr/bin/env bash
# Predict with a single trained nnU-Net v2 fold (NOT an ensemble).
set -euo pipefail
if [[ $# -lt 3 || $# -gt 4 ]]; then
  echo "Usage: bash scripts/predict.sh DATASET_ID IMAGE_DIR OUTPUT_DIR [FOLD]" >&2
  exit 2
fi
: "${nnUNet_raw:?Set nnUNet_raw}"
: "${nnUNet_preprocessed:?Set nnUNet_preprocessed}"
: "${nnUNet_results:?Set nnUNet_results}"
DATASET="$1"
INPUT="$2"
OUTPUT="$3"
FOLD="${4:-0}"
[[ -d "$INPUT" ]] || { echo "Image directory not found: $INPUT" >&2; exit 1; }
mkdir -p "$OUTPUT"
nnUNetv2_predict -d "$DATASET" -c 3d_fullres -f "$FOLD" -i "$INPUT" -o "$OUTPUT"
