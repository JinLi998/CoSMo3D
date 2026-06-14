#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"

eval "$(conda shell.bash hook)"
conda activate find3d

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python -m vis_code.seg_and_vis \
  --checkpoint_path "dataset/checkpoints/ours_final.pth" \
  --data_path "data_test/coarse_b'29_0cb'" \
  --mesh_path "data_test/29_0cb.glb" \
  --output_dir "results" \
  --category "vase" \
  --sample_name "29_0cb"
