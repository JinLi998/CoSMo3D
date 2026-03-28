import torch
import torch.nn as nn


class CanonicalColorLoss(nn.Module):
    """Chamfer-based canonical color consistency loss."""

    def __init__(self):
        super().__init__()

    @staticmethod
    def chamfer_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if x.size(0) == 0 or y.size(0) == 0:
            return torch.tensor(0.0, device=x.device)
        dist_matrix = torch.cdist(x, y, p=2)
        min_dist_x = torch.min(dist_matrix, dim=1)[0].mean()
        min_dist_y = torch.min(dist_matrix, dim=0)[0].mean()
        return (min_dist_x + min_dist_y) / 2.0

    def forward(self, canoncolor_out, gt_color, pt_offset, mask_pts):
        device = canoncolor_out.device
        batch_size = len(pt_offset) - 1
        total_loss = 0.0
        count = 0

        obj_start_idxs = torch.cat((torch.tensor([0], device=device), pt_offset[:-1]))
        for obj_idx in range(batch_size):
            start_idx = obj_start_idxs[obj_idx]
            end_idx = pt_offset[obj_idx]
            if end_idx - start_idx == 0:
                continue

            obj_mask_pts = mask_pts[obj_idx].to(device)
            part_losses = []
            for mask_idx in range(obj_mask_pts.size(0)):
                part_indices = torch.nonzero(obj_mask_pts[mask_idx]).squeeze(1)
                if part_indices.numel() < 2:
                    continue
                global_indices = start_idx + part_indices
                pred_colors = canoncolor_out[global_indices]
                true_colors = gt_color[global_indices]
                part_losses.append(self.chamfer_distance(pred_colors, true_colors))

            if part_losses:
                total_loss += torch.mean(torch.stack(part_losses))
                count += 1

        if count == 0:
            return torch.tensor(0.0, device=device)
        return total_loss / count
