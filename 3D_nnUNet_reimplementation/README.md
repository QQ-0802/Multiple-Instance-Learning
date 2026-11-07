# Cervical lymph-node segmentation: nnU-Net v2 reimplementation

**Important provenance note:** This is an independently authored, executable
nnU-Net v2 workflow, NOT source code recovered from the Neusoft environment
used in the original LSCC study. It has not produced the study's published
segmentations or Dice score. Do not describe it as the original study code.

Upstream nnU-Net: https://github.com/MIC-DKFZ/nnUNet
Official dataset format: https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/reference/dataset-format.md

## What this project implements

- `scripts/prepare_dataset.py`: convert NIfTI CECT and binary expert LN masks
  to nnU-Net v2 dataset layout, without adding external masks to training.
- `scripts/make_splits.py`: patient-grouped training/validation splits.
- `scripts/train.sh`: official nnU-Net v2 planning and 3D training.
- `scripts/predict.sh`: official nnU-Net v2 inference.
- `scripts/evaluate.py`: Dice and symmetric HD95 in mm.
- `scripts/components.py`: optional individual connected-component LN candidates.
- `tests/test_workflow.py`: synthetic-data tests for helper scripts.

## Setup

Install hardware-compatible PyTorch (https://pytorch.org/get-started/locally/)
and then:

```bash
python -m pip install -r requirements.txt
export nnUNet_raw="/absolute/path/nnUNet_raw"
export nnUNet_preprocessed="/absolute/path/nnUNet_preprocessed"
export nnUNet_results="/absolute/path/nnUNet_results"
mkdir -p "$nnUNet_raw" "$nnUNet_preprocessed" "$nnUNet_results"
```

All case IDs and patient IDs must be pseudonymized. Never upload patient CT,
manual masks, raw clinical metadata, local manifest CSVs, or model weights
without permission.

## Input manifest

Create a **private** CSV with the columns shown in `manifest_template.csv`.
`split` must be `train`, `external_1`, or `external_2`. Images and masks must
be 3D `.nii.gz` volumes sharing shape and affine. Training reference masks
must contain only 0=background and 1=cervical lymph node. One patient's
volumes must not cross split boundaries. The classification dataset's
218/174/81 sample sizes are NOT assumed to equal segmentation training sizes.

## Prepare original NIfTI images and masks

```bash
python scripts/prepare_dataset.py --manifest /private/manifest.csv \
  --raw-root "$nnUNet_raw" --dataset-id 701
```

Creates `Dataset701_CervicalLymphNodes/{dataset.json,imagesTr,labelsTr,imagesTs}`.
Original CT intensity values are preserved by default. nnU-Net's CT channel
normalization applies during its own planning. Optionally use
`--clip-low HU_LOW --clip-high HU_HIGH` only if the ORIGINAL study windowing
parameters have been verified; do not invent them.

Reference masks of external cases are never written to `labelsTr`.

## Plan + train with patient-grouped splits

Run these steps IN THIS ORDER:

```bash
nnUNetv2_plan_and_preprocess -d 701 -c 3d_fullres --verify_dataset_integrity
python scripts/make_splits.py --manifest /private/manifest.csv \
  --preprocessed-dir "$nnUNet_preprocessed/Dataset701_CervicalLymphNodes" \
  --folds 5 --seed 42
bash scripts/train.sh 701 0
```

To run all five folds, train folds 0, 1, 2, 3 and 4 separately. The group
split is generated exclusively from `train` records. External data must not
be used to fit preprocessing, choose checkpoints or tune postprocessing.

## Predict + evaluate

```bash
bash scripts/predict.sh 701 "$nnUNet_raw/Dataset701_CervicalLymphNodes/imagesTs" \
  /private/predictions 0

python scripts/evaluate.py --pred-dir /private/predictions \
  --ref-dir /private/external_reference_masks \
  --output-csv /private/segmentation_metrics.csv

python scripts/components.py --pred-dir /private/predictions \
  --output-dir /private/ln_candidates --min-volume-mm3 10
```

`predict.sh` defaults to **one trained fold**, not a five-fold ensemble.
External reference masks must be supplied as independent `{case_id}.nii.gz`
files in a private directory. Dice=1 and HD95=0 for two empty masks; for a
single empty mask, Dice=0 and HD95 is undefined.

## Test the utility functions (does NOT run GPU training)

```bash
python -m unittest discover -s tests -v
```

## Limitations and study-specific dependencies

This folder is a **transparent reimplementation** using nnU-Net v2.
The original LSCC manuscript describes nnU-Net segmentation executed
in a separate Neusoft-provided environment, with later manual corrections
using ITK-SNAP. The original Neusoft execution scripts, nnU-Net generation
(v1/v2), original preprocessing/window parameters, trained checkpoints,
train/validation split, software versions, manual mask correction protocol,
and reference CTs were not supplied for this reimplementation.

The project cannot establish reproduction of the original published Dice=0.75
or the original masks until the original study material is obtained and
the reproduction experiment is actually performed and reported.

OnekeyAI does NOT provide the 3D nnU-Net implementation in this folder.
The downstream MIL, LightGBM, evaluation and Grad-CAM study notebooks
are maintained separately in:
https://github.com/QQ-0802/Multiple-Instance-Learning
and depend on https://github.com/OnekeyAI-Platform/onekey .

## Recommended citation

Isensee F, Jaeger PF, Kohl SAA, Petersen J, Maier-Hein KH. nnU-Net:
a self-configuring method for deep learning-based biomedical image
segmentation. Nature Methods. 2021;18:203-211.

Record the actual experiment versions and configurations in
`RUN_RECORD_TEMPLATE.md` rather than claiming unverified equivalence.
