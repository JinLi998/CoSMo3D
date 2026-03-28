#!/usr/bin/env bash
set -euo pipefail

# Example release training command.
# Edit GPU IDs and paths for your environment.

GPU_IDS=${GPU_IDS:-0,1,2,3,4,5,6,7}
DATA_ROOT=${DATA_ROOT:-dataset/trainingdata}
CKPT_DIR=${CKPT_DIR:-results/find3d_d3compat_release}
PRETRAINED=${PRETRAINED:-dataset/checkpoints/orgfind3d.pth}

echo "GPU_IDS=${GPU_IDS}"
echo "DATA_ROOT=${DATA_ROOT}"
echo "CKPT_DIR=${CKPT_DIR}"
echo "PRETRAINED=${PRETRAINED}"

CUDA_VISIBLE_DEVICES="${GPU_IDS}" python -m train_code.train_release \
  --data_root "${DATA_ROOT}" \
  --ckpt_dir "${CKPT_DIR}" \
  --pretrained_path "${PRETRAINED}" \
  --n_epoch 200 \
  --batch_size 32 \
  --lr 0.0005 \
  --eta_min 0.00005 \
  --canoncolor_loss_weight 0.2 \
  --bbox_loss_weight 5.0 \
  --drop_canoncolor_last_n_epochs 30
