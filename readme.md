# CoSMo3D: Open-World Promptable 3D Semantic Segmentation through LLM-Guided Canonical Spatial Modeling

**CVPR 2026 — Full Score**

**Paper**: [arXiv](https://arxiv.org/pdf/2603.01205)

![Teaser](images/teaser.png)

Open-world promptable 3D semantic segmentation remains brittle as semantics are inferred in the input sensor coordinates. Yet humans interpret parts via functional roles in a canonical space – wings extend laterally, handles protrude to the side, and legs support from below. To fill this gap, we propose **CoSMo3D**, which attains canonical space perception by inducing a latent canonical reference frame learned directly from data. By construction, we create a unified canonical dataset through LLM-guided intra- and cross-category alignment, exposing canonical spatial regularities across 200 categories. By induction, we realize canonicality through a dual-branch architecture with canonical map anchoring and canonical box calibration, collapsing pose variation and symmetry into a stable canonical embedding. This shift from input pose space to canonical embedding yields far more stable and transferable part semantics. CoSMo3D establishes new state of the art in open-world promptable 3D segmentation.

<!-- ![Framework](images/framework.png) -->



---

## Environment

- **CUDA 12.2**:  
  `conda create --name cosmo3d --file environment.txt`
- **Other CUDA versions**:  
  `conda env create -f environment.yml`

Then activate the environment: `conda activate cosmo3d`.

---

## Pretrained Model

Download the pretrained CoSMo3D model and save it locally:

- **Download**: [ours_final.pth](https://huggingface.co/PrinterLi/CoSMo3D/blob/main/ours_final.pth) (Hugging Face)
- **Save to**: `dataset/checkpoints/ours_final.pth`

Create the directory if needed: `mkdir -p dataset/checkpoints`, then place the downloaded file there.

---

## Datasets (download)

Place archives under `dataset/` and extract there so paths match the defaults used by `eval_benchmark` and `train_code`.

### Test benchmarks (Hugging Face)

| Archive | Resolve URL (direct download) | Default path after extract |
| --- | --- | --- |
| 3DCompat200 test | [test_3dcompat200.tar.gz](https://huggingface.co/PrinterLi/CoSMo3D/resolve/main/test_3dcompat200.tar.gz) | `dataset/test_3dcompat200/` |
| ShapeNetPart test | [test_shapenetpart.tar.gz](https://huggingface.co/PrinterLi/CoSMo3D/resolve/main/test_shapenetpart.tar.gz) | `dataset/test_shapenetpart/` |

Example (Linux / macOS; from repo root):

```bash
mkdir -p dataset
cd dataset
curl -L -O https://huggingface.co/PrinterLi/CoSMo3D/resolve/main/test_3dcompat200.tar.gz
curl -L -O https://huggingface.co/PrinterLi/CoSMo3D/resolve/main/test_shapenetpart.tar.gz
tar -xzf test_3dcompat200.tar.gz
tar -xzf test_shapenetpart.tar.gz
cd ..
```

If the tarball contains a single top-level folder, rename or move it so the eval script sees `dataset/test_3dcompat200` and `dataset/test_shapenetpart` as roots. **3DCompat200** evaluation expects splits `seenclass`, `unseen`, and `shapenetpart` under that root (see `EvalData3D` in `model/data/data.py`). **ShapeNetPart** expects `*test*.h5` files directly under `dataset/test_shapenetpart` (or set `--data_root` accordingly).

**PartNetE** is not included in the archives above. Prepare it under `dataset/partnet` following `model/evaluation/benchmark/README.md` (`test/` plus `PartNetE_meta.json`). The release eval defaults to `--data_root dataset/partnet` for the partnet benchmark.

### Training data (Hugging Face)

| Archive | Resolve URL | Default `--data_root` |
| --- | --- | --- |
| Training pack | [trainingdata.tar.gz](https://huggingface.co/PrinterLi/CoSMo3D/resolve/main/trainingdata.tar.gz) (~2.28 GB) | `dataset/trainingdata/` |

File pages on Hugging Face (same files): [test_3dcompat200.tar.gz](https://huggingface.co/PrinterLi/CoSMo3D/blob/main/test_3dcompat200.tar.gz), [test_shapenetpart.tar.gz](https://huggingface.co/PrinterLi/CoSMo3D/blob/main/test_shapenetpart.tar.gz), [trainingdata.tar.gz](https://huggingface.co/PrinterLi/CoSMo3D/blob/main/trainingdata.tar.gz).

Example:

```bash
mkdir -p dataset
cd dataset
curl -L -O https://huggingface.co/PrinterLi/CoSMo3D/resolve/main/trainingdata.tar.gz
tar -xzf trainingdata.tar.gz
cd ..
```

Point `train_code.train_release` at the folder that contains `train.txt` (often `dataset/trainingdata` after extraction). If the archive uses another top-level name, pass that path to `--data_root`.

---

## Quick Example

1. Create a `results` folder:  
   `mkdir results`
2. Run the single-sample evaluation:  
   `python -m app.segment.eval_benchmark`

---

## Benchmark Evaluation (Release)

The release benchmark script is located at `eval_benchmark/eval_benchmark.py`. Default roots match the Hugging Face test archives under `dataset/` (see [Datasets (download)](#datasets-download)).

- Evaluate **3dcompat200** (default: `dataset/test_3dcompat200`):  
  `python eval_benchmark/eval_benchmark.py --benchmark 3dcompat200 --split seenclass --checkpoint_path dataset/checkpoints/ours_final.pth`
- Evaluate **partnet** (default: `dataset/partnet`):  
  `python eval_benchmark/eval_benchmark.py --benchmark partnet --checkpoint_path dataset/checkpoints/ours_final.pth --save_dir results/partnet`
- Evaluate **shapenetpart** (default: `dataset/test_shapenetpart`):  
  `python eval_benchmark/eval_benchmark.py --benchmark shapenetpart --checkpoint_path dataset/checkpoints/ours_final.pth`

Optional flags:

- `--data_root <path>`: override default benchmark path.
- `--rotate`: apply predefined random rotations for partnet/shapenetpart.
- `--subset`: evaluate predefined subsets when supported.
- `--plain_prompt`: use plain part names instead of decorated prompts.

---

## Training (Release)

Release training uses `train_code/train_release.py`. It trains the canonical-align + bbox model (`release_module/network/canoncolor_bbox_pre.py`) with a baseline contrastive loss, canonical color loss, and bbox loss. Canonical color loss is dropped automatically in the **last N epochs** (default 30).

### Data layout

- Set `--data_root` to a folder that contains `train.txt`, where each line is an absolute or relative path to one training sample directory.
- Each sample directory should contain: `mask_labels.txt`, `mask2points.pt`, `points.pt`, `normals.pt`, `rgb.pt`. Text features are cached per sample as `text_feat.pt` when missing.

### Single command (from repo root)

```bash
python -m train_code.train_release \
  --data_root dataset/trainingdata \
  --ckpt_dir results/find3d_release \
  --pretrained_path dataset/checkpoints/orgfind3d.pth \
  --n_epoch 200 \
  --batch_size 32 \
  --lr 0.0005 \
  --eta_min 0.00005 \
  --canoncolor_loss_weight 0.2 \
  --bbox_loss_weight 5.0 \
  --drop_canoncolor_last_n_epochs 30
```

### Multi-GPU

The script uses all visible CUDA devices. Example:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m train_code.train_release --data_root dataset/trainingdata --ckpt_dir results/find3d_release --pretrained_path dataset/checkpoints/orgfind3d.pth
```

### Shell helper

`train_code/run_train_release.sh` wraps the same defaults; override paths with `GPU_IDS`, `DATA_ROOT`, `CKPT_DIR`, and `PRETRAINED`.

### Optional flags

- `--continue_path <ckpt.pth>`: resume optimizer and scheduler when present in the checkpoint.
- `--category_alignment_json` / `--shape_category_json`: optional JSON files for per-category rotation alignment before augmentation; if omitted, alignment is skipped.
- `--num_workers`, `--save_every`, `--log_every`, `--ddp_port`: DataLoader workers, checkpoint interval, log interval, and DDP port.

Training data and loss helpers live under `release_module/training/` (`data_release.py`, `loss_canonical_color.py`, `loss_bbox.py`).

---

## Todo List

- [ ] Release example test & training models
- [ ] Release all test datasets and corresponding test code
- [ ] Release training code and training datasets
