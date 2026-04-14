"""
Release dataset utilities for training.

This module contains only what is required by train_release.py:
- TrainingData
- collate_fn

Training samples are expected to be already posed in the canonical / normalized frame
distributed in the released training pack; no extra rotation alignment is applied here.
"""

import os
from collections.abc import Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
from transformers import AutoModel, AutoTokenizer

from model.data.augmentation import (
    Add,
    CenterShift,
    ChromaticAutoContrast,
    ChromaticJitter,
    ChromaticTranslation,
    Collect,
    GridSample,
    NormalizeColor,
    RandomFlip,
    RandomJitter,
    RandomRotate,
    RandomScale,
    ToTensor,
)


def _normalize_to_rgb(points: torch.Tensor) -> torch.Tensor:
    min_vals, _ = torch.min(points, dim=0)
    max_vals, _ = torch.max(points, dim=0)
    ranges = torch.where(
        (max_vals - min_vals) < 1e-8,
        torch.tensor(1e-8, device=points.device),
        (max_vals - min_vals),
    )
    return torch.clamp((points - min_vals) / ranges, 0.0, 1.0)


def prep_points_train(xyz, rgb, normal, mask2pt, canonical_color=None):
    xyz_change_axis = np.concatenate(
        [-xyz[:, 0:1], xyz[:, 2:3], xyz[:, 1:2]],
        axis=1,
    )
    data_dict = {
        "coord": xyz_change_axis,
        "color": rgb,
        "normal": normal,
        "mask2pt": mask2pt,
        "canoncial_color": canonical_color,
    }
    data_dict = CenterShift(apply_z=True)(data_dict)
    data_dict = RandomScale(scale=[0.8, 1.2], anisotropic=True)(data_dict)
    data_dict = RandomRotate(angle=[-1, 1], axis="z", center=[0, 0, 0], p=1)(data_dict)
    data_dict = RandomRotate(angle=[-1, 1], axis="x", p=1)(data_dict)
    data_dict = RandomRotate(angle=[-1, 1], axis="y", p=1)(data_dict)
    data_dict = RandomFlip(p=0.5)(data_dict)
    data_dict = RandomJitter(sigma=0.005, clip=0.02)(data_dict)
    data_dict = ChromaticAutoContrast(p=0.2, blend_factor=None)(data_dict)
    data_dict = ChromaticTranslation(p=0.95, ratio=0.05)(data_dict)
    data_dict = ChromaticJitter(p=0.95, std=0.05)(data_dict)
    data_dict = GridSample(grid_size=0.02, hash_type="fnv", mode="train", return_grid_coord=True)(data_dict)
    data_dict = CenterShift(apply_z=False)(data_dict)
    data_dict = NormalizeColor()(data_dict)
    data_dict = Add(keys_dict=dict(condition="S3DIS"))(data_dict)
    data_dict = ToTensor()(data_dict)
    data_dict = Collect(
        keys=("coord", "grid_coord", "mask2pt", "canoncial_color"),
        offset_keys_dict={"offset": "coord", "mask_offset": "mask2pt"},
        feat_keys=("color", "normal"),
    )(data_dict)
    return data_dict


def collate_fn(batch):
    if not isinstance(batch, Sequence):
        raise TypeError(f"{type(batch)} is not supported.")

    if isinstance(batch[0], torch.Tensor):
        if len(batch) > 1:
            try:
                return torch.cat(list(batch))
            except Exception:
                return list(batch)
        return batch[0]

    if isinstance(batch[0], str):
        return list(batch)

    if isinstance(batch[0], Sequence):
        if isinstance(batch[0][0], str):
            return batch
        for data in batch:
            data.append(torch.tensor([data[0].shape[0]]))
        batch = [collate_fn(samples) for samples in zip(*batch)]
        batch[-1] = torch.cumsum(batch[-1], dim=0).int()
        return batch

    if isinstance(batch[0], Mapping):
        batch_new = {key: collate_fn([d[key] for d in batch]) for key in batch[0] if key != "mask2pt"}
        if "mask2pt" in batch[0]:
            batch_new["mask2pt"] = [d["mask2pt"] for d in batch]
        for key in batch_new.keys():
            if "offset" in key:
                batch_new[key] = torch.cumsum(batch_new[key], dim=0)
        return batch_new

    return default_collate(batch)


class TrainingData(Dataset):
    """Loads samples listed in ``{data_root}/train.txt`` (canonical-frame training pack)."""

    def __init__(self, data_root: str):
        self.data_root = data_root
        with open(f"{data_root}/train.txt", "r", encoding="utf-8") as f:
            self.obj_path_list = f.read().splitlines()

        self.model = AutoModel.from_pretrained("google/siglip-base-patch16-224")
        self.tokenizer = AutoTokenizer.from_pretrained("google/siglip-base-patch16-224")

    def __getitem__(self, item):
        data_dir = self.obj_path_list[item]
        with open(f"{data_dir}/mask_labels.txt", "r", encoding="utf-8") as f:
            labels = f.read().splitlines()

        mask_pts = torch.load(f"{data_dir}/mask2points.pt", map_location="cpu")
        pts_xyz = torch.load(f"{data_dir}/points.pt", map_location="cpu")
        normal = torch.load(f"{data_dir}/normals.pt", map_location="cpu")
        pts_rgb = torch.load(f"{data_dir}/rgb.pt", map_location="cpu") * 255

        canonical_color = _normalize_to_rgb(pts_xyz)

        point_dict = prep_points_train(
            pts_xyz.numpy(),
            pts_rgb.numpy(),
            normal.numpy(),
            mask_pts.numpy(),
            canonical_color.numpy(),
        )
        point_dict["labels"] = labels

        text_feat_path = f"{data_dir}/text_feat.pt"
        if os.path.exists(text_feat_path):
            text_feat = torch.load(text_feat_path, map_location="cpu")
        else:
            inputs = self.tokenizer(labels, padding="max_length", truncation=True, return_tensors="pt")
            with torch.no_grad():
                text_feat = self.model.get_text_features(**inputs)
            text_feat = text_feat / (text_feat.norm(dim=-1, keepdim=True) + 1e-12)
            torch.save(text_feat, text_feat_path)

        point_dict["label_embeds"] = text_feat
        return point_dict

    def __len__(self):
        return len(self.obj_path_list)
