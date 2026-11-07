#!/usr/bin/env bash
# Call after nnUNetv2_plan_and_preprocess and (optional) make_splits.py.
set -euo pipefail
if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: bash scripts/train.sh DATASET_ID [FOLD]" >&2
  exit 2
fi
: "${nnUNet_raw:?Set nnUNet_raw}"
: "${nnUNet_preprocessed:?Set nnUNet_preprocessed}"
: "${nnUNet_results:?Set nnUNet_results}"
DATASET="$1"
FOLD="${2:-0}"
nnUNetv2_train "$DATASET" 3d_fullres "$FOLD" --npz
