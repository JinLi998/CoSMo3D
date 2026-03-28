from typing import List

import torch
import torch.nn as nn


class PointBasedBBoxOffsetLoss(nn.Module):
    """
    Bounding-box offset regression loss on part masks.
    Uses mm-scaled smooth-L1 style formulation for stable gradients.
    """

    def __init__(self, beta_mm: float = 10.0, debug: bool = False, min_point_count: int = 1):
        super().__init__()
        self.scale = 1000.0
        self.beta = beta_mm
        self.debug = debug
        self.min_point_count = min_point_count

    def forward(
        self,
        bbox_pred: torch.Tensor,
        pts: torch.Tensor,
        pt_offset: torch.Tensor,
        mask_points: List[torch.Tensor],
    ) -> torch.Tensor:
        num_objects = pt_offset.shape[0]
        if len(mask_points) != num_objects:
            raise ValueError(f"mask_points length mismatch: {len(mask_points)} vs {num_objects}")
        if bbox_pred.shape[0] == 0:
            return torch.tensor(0.0, device=bbox_pred.device)

        device = pts.device
        point_offsets = torch.cat([torch.tensor([0], device=device, dtype=pt_offset.dtype), pt_offset.to(device)])
        obj_ids = torch.searchsorted(point_offsets[1:], torch.arange(pts.shape[0], device=device), right=False)

        obj_sums = torch.zeros((num_objects, 3), device=device)
        obj_counts = torch.zeros(num_objects, device=device)
        ones = torch.ones((pts.shape[0],), device=device)
        obj_sums.scatter_add_(0, obj_ids.unsqueeze(1).repeat(1, 3), pts)
        obj_counts.scatter_add_(0, obj_ids, ones)
        object_centers = obj_sums / obj_counts.clamp(min=1.0).unsqueeze(1)

        gt_offsets_list = []
        valid_preds_list = []
        part_idx = 0

        for obj_idx in range(num_objects):
            obj_masks = mask_points[obj_idx].to(device)
            num_parts = obj_masks.shape[0]
            if num_parts == 0:
                continue

            pred_start = part_idx
            pred_end = part_idx + num_parts
            if pred_end > bbox_pred.shape[0]:
                raise ValueError(f"prediction index out of range: {pred_start}-{pred_end} > {bbox_pred.shape[0]}")
            obj_preds = bbox_pred[pred_start:pred_end]
            part_idx = pred_end

            obj_pts = pts[point_offsets[obj_idx]:point_offsets[obj_idx + 1]]
            obj_center = object_centers[obj_idx]
            part_mask_bool = obj_masks > 0
            point_counts = part_mask_bool.sum(dim=1)

            part_masks = part_mask_bool.unsqueeze(-1)
            masked_pts = obj_pts.unsqueeze(0) * part_masks
            part_min = torch.min(torch.where(part_masks, masked_pts, torch.inf), dim=1)[0]
            part_max = torch.max(torch.where(part_masks, masked_pts, -torch.inf), dim=1)[0]

            valid_parts = (point_counts >= self.min_point_count) & ~torch.any(torch.isinf(part_min), dim=1)
            if valid_parts.sum().item() == 0:
                continue

            valid_gt = torch.cat([part_min[valid_parts] - obj_center, part_max[valid_parts] - obj_center], dim=1)
            valid_pred = obj_preds[valid_parts]
            gt_offsets_list.append(valid_gt)
            valid_preds_list.append(valid_pred)

        if not gt_offsets_list:
            return torch.tensor(0.0, device=device)

        gt_offsets = torch.cat(gt_offsets_list, dim=0)
        valid_bbox_pred = torch.cat(valid_preds_list, dim=0)
        if valid_bbox_pred.shape[0] != gt_offsets.shape[0]:
            raise ValueError(f"filtered parts mismatch: {valid_bbox_pred.shape[0]} vs {gt_offsets.shape[0]}")

        diff = (valid_bbox_pred - gt_offsets) * self.scale
        abs_diff = torch.abs(diff)
        per_dim_loss = torch.where(
            abs_diff <= self.beta,
            0.5 * (abs_diff / self.beta) ** 2,
            (abs_diff / self.beta) - 0.5,
        )
        return per_dim_loss.mean(dim=1).mean()
