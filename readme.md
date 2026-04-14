# CoSMo3D: Open-World Promptable 3D Semantic Segmentation through LLM-Guided Canonical Spatial Modeling

## Todo List

- [x] Release example test path and pretrained checkpoint ([Quick Example](#quick-example), [ours_final.pth](https://huggingface.co/PrinterLi/CoSMo3D/blob/main/ours_final.pth))
- [x] Release benchmark test data (HF) and evaluation code (`eval_benchmark/`)
- [x] Release training code and training data (`train_code/`, `release_module/training/`, [trainingdata.tar.gz](https://huggingface.co/PrinterLi/CoSMo3D/blob/main/trainingdata.tar.gz))
- [ ] Release canonical / normalized meshes in a standard format for paper-quality figures

---

**CVPR 2026 — Full Score**

**Paper**: [arXiv](https://arxiv.org/pdf/2603.01205) | **Project Page**: [CoSMo3D](https://jinli998.github.io/cosmo3d_page/)

![Teaser](images/teaser.png)



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

## Quick Example

1. Create a `results` folder:  
   `mkdir results`
2. Run the single-sample evaluation:  
   `python -m app.segment.eval_benchmark`

---

## Datasets (download)

Place archives under `dataset/` and extract there so paths match the defaults used by `eval_benchmark` and `train_code`.

### Test benchmarks (Hugging Face)

| Archive | Resolve URL (direct download) | Default path after extract |
| --- | --- | --- |
| 3DCompat200 test | [test_3dcompat200.tar.gz](https://huggingface.co/PrinterLi/CoSMo3D/resolve/main/test_3dcompat200.tar.gz) | `dataset/test_3dcompat200/` |
| ShapeNetPart test | [test_shapenetpart.tar.gz](https://huggingface.co/PrinterLi/CoSMo3D/resolve/main/test_shapenetpart.tar.gz) | `dataset/test_shapenetpart/` |
| PartNetE test | [data/test.zip](https://huggingface.co/datasets/minghua/PartSLIP/resolve/main/data/test.zip) (~9.4 GB, from PartSLIP) | `dataset/partnet/test/` after unzip |
| PartNetE metadata | [PartNetE_meta.json](https://huggingface.co/datasets/minghua/PartSLIP/resolve/main/PartNetE_meta.json) | `dataset/partnet/PartNetE_meta.json` |

PartNetE files are hosted in the [minghua/PartSLIP](https://huggingface.co/datasets/minghua/PartSLIP/tree/main/data) dataset ([`data/` folder](https://huggingface.co/datasets/minghua/PartSLIP/tree/main/data)). Download `test.zip` and place `PartNetE_meta.json` in the **same** root as the unzipped `test/` directory so the layout is `dataset/partnet/test/<Category>/...` and `dataset/partnet/PartNetE_meta.json` (matches `EvalPartNetE` in `model/data/data.py`). Metadata file page: [PartNetE_meta.json](https://huggingface.co/datasets/minghua/PartSLIP/blob/main/PartNetE_meta.json).

Example (Linux / macOS; from repo root):

```bash
mkdir -p dataset/partnet
cd dataset/partnet
curl -L -O https://huggingface.co/datasets/minghua/PartSLIP/resolve/main/data/test.zip
curl -L -O https://huggingface.co/datasets/minghua/PartSLIP/resolve/main/PartNetE_meta.json
unzip -q test.zip   # should produce test/ with per-category subfolders
cd ../..
```

```bash
mkdir -p dataset
cd dataset
curl -L -O https://huggingface.co/PrinterLi/CoSMo3D/resolve/main/test_3dcompat200.tar.gz
curl -L -O https://huggingface.co/PrinterLi/CoSMo3D/resolve/main/test_shapenetpart.tar.gz
tar -xzf test_3dcompat200.tar.gz
tar -xzf test_shapenetpart.tar.gz
cd ..
```

If the tarball contains a single top-level folder, rename or move it so the eval script sees `dataset/test_3dcompat200` and `dataset/test_shapenetpart` as roots.

Run **3DCompat200** as two subsets:
- **coarse** subset: instances named `coarse_*`
- **fine** subset: instances named `fine_*`

The 3DCompat200 root layout should be:  
`dataset/test_3dcompat200/<category>/{coarse_*|fine_*}/...`  
**ShapeNetPart** root should be in `shapenetpart_hdf5_2048` format and contain `*test*.h5` directly under the root directory (or set `--data_root` accordingly).  
**PartNetE** root should contain `test/` and `PartNetE_meta.json` together. For reproducible random rotations on PartNetE, see `model/evaluation/benchmark/README.md` and `model/evaluation/benchmark/benchmark_reproducibility/partnete/`.

### Training data (Hugging Face)

| Archive | Resolve URL | Default `--data_root` |
| --- | --- | --- |
| Training pack | [trainingdata.tar.gz](https://huggingface.co/PrinterLi/CoSMo3D/resolve/main/trainingdata.tar.gz) (~2.28 GB) | `dataset/trainingdata/` |

The released **training** pack is **already canonicalized / normalized** (poses aligned to a consistent canonical frame during dataset construction). Training code applies only the on-the-fly augmentations in `release_module/training/data_release.py` (axis remap, scaling, rotation jitter, etc.).

File pages on Hugging Face (browser): CoSMo3D [test_3dcompat200.tar.gz](https://huggingface.co/PrinterLi/CoSMo3D/blob/main/test_3dcompat200.tar.gz), [test_shapenetpart.tar.gz](https://huggingface.co/PrinterLi/CoSMo3D/blob/main/test_shapenetpart.tar.gz), [trainingdata.tar.gz](https://huggingface.co/PrinterLi/CoSMo3D/blob/main/trainingdata.tar.gz); PartSLIP [data/test.zip](https://huggingface.co/datasets/minghua/PartSLIP/blob/main/data/test.zip), [PartNetE_meta.json](https://huggingface.co/datasets/minghua/PartSLIP/blob/main/PartNetE_meta.json).

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

## Benchmark Evaluation 

The release benchmark script is located at `eval_benchmark/eval_benchmark.py`. Default roots match the Hugging Face test archives under `dataset/` (see [Datasets (download)](#datasets-download)).

- Evaluate **3dcompat200 coarse**:  
  `python eval_benchmark/eval_benchmark.py --benchmark d3compat --d3com_datatype coarse --checkpoint_path dataset/checkpoints/ours_final.pth`
- Evaluate **3dcompat200 fine** :  
  `python eval_benchmark/eval_benchmark.py --benchmark d3compat --d3com_datatype fine --checkpoint_path dataset/checkpoints/ours_final.pth`
- Evaluate **partnete** :  
  `python eval_benchmark/eval_benchmark.py --benchmark partnete --checkpoint_path dataset/checkpoints/ours_final.pth --save_dir results/partnet`
- Evaluate **shapenetpart** :  
  `python eval_benchmark/eval_benchmark.py --benchmark shapnetpart --checkpoint_path dataset/checkpoints/ours_final.pth`

Optional flags:

- `--data_root <path>`: override default benchmark path.
- `--d3com_datatype {coarse,fine}`: select d3compat subset when using default path.
- `--rotate`: apply predefined random rotations for partnet/shapenetpart.
- `--canonical`: force no rotation.
- `--subset`: evaluate predefined subsets when supported.
- `--plain_prompt`: use plain part names instead of decorated prompts.
- `--part_query`: use plain part names.

---

## Training 

Release training uses `train_code/train_release.py`. It trains the canonical-align + bbox model (`release_module/network/canoncolor_bbox_pre.py`) with a baseline contrastive loss, canonical color loss, and bbox loss. Canonical color loss is dropped automatically in the **last N epochs** (default 30).

### Training checkpoint

- **Download**: [orgfind3d.pth](https://huggingface.co/PrinterLi/CoSMo3D/blob/main/orgfind3d.pth)
- **Save to**: `dataset/checkpoints/orgfind3d.pth`

### Data layout

- Set `--data_root` to a folder that contains `train.txt`, where each line is an absolute or relative path to one training sample directory.
- Each sample directory should contain: `mask_labels.txt`, `mask2points.pt`, `points.pt`, `normals.pt`, `rgb.pt`. Text features are cached per sample as `text_feat.pt` when missing.
- Use the Hugging Face **trainingdata** archive above: point clouds are **pre-canonicalized**; do not expect the loader to fix global orientation from metadata files.

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
- `--num_workers`, `--save_every`, `--log_every`, `--ddp_port`: DataLoader workers, checkpoint interval, log interval, and DDP port.

Training data and loss helpers live under `release_module/training/` (`data_release.py`, `loss_canonical_color.py`, `loss_bbox.py`).
