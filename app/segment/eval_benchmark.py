"""
Single-sample evaluation script.
  python -m app.segment.eval_benchmark

In addition to 3D data (point cloud, etc.), a mask_labels.txt file is required containing
segmentation text labels, e.g.:
  base
  plant
  body
  neck

mask2points.pt is not actually used; an empty array can be used as substitute.
"""

import os
import json
import torch
import argparse
import numpy as np
from transformers import AutoTokenizer, AutoModel

from app.segment._data import prep_points_val3d
from model.evaluation.utils import load_model
from release_module.network.canoncolor_bbox_pre import PointSemSegWithDecoder

# Required external function/class dependencies (core logic preserved)
def get_part_labels(pts_xyz, mask_pts):
    """
    Assign part labels to point cloud based on mask_pts (fix device mismatch + preserve original logic).
    Args:
        pts_xyz: Point cloud tensor [5000, 3]
        mask_pts: Part mask tensor [K, 5000] (K is number of parts)
    Returns:
        labels: Per-point part label tensor [5000,] (-1 for unmatched, otherwise part index)
    """
    # 1. Get input tensor device (key: all internal tensors stay on same device as input)
    device = pts_xyz.device
    
    # 2. Dimension validation (preserve original assertion logic)
    assert pts_xyz.shape == (5000, 3), f"pts_xyz shape should be [5000, 3], got {pts_xyz.shape}"
    K, N = mask_pts.shape
    assert N == 5000, f"mask_pts second dim should be 5000, got {N}"
    
    # 3. Explicitly specify device when creating/converting tensors (core fix)
    mask_int = mask_pts.to(torch.int32).to(device)  # Ensure on target device
    num_activated = mask_int.sum(dim=0)
    
    # labels initialization: specify device, keep consistent with input
    labels = torch.full((5000,), -1, dtype=torch.long, device=device)
    valid_mask = num_activated == 1  # Auto-inherit mask_int device
    
    if valid_mask.any():
        # Sliced tensors remain on same device
        valid_mask_slice = mask_int[:, valid_mask].to(torch.int32)  # Already on target device, to() only ensures type
        valid_indices = torch.argmax(valid_mask_slice, dim=0)  # Auto on target device
        labels[valid_mask] = valid_indices  # Device fully matches, no errors
    
    return labels

# Core evaluation function (simplified)
def compute_3d_iou_single(net_out, text_embeds, temperature, cat, xyz_sub, xyz_full, gt_full, 
                        save_path, N_CHUNKS=1):
    # -------------------------- 1. Compute prediction labels (preserve original logic) --------------------------
    # Compute text-point cloud similarity score
    logits = net_out @ text_embeds.t() * temperature
    # Prediction labels (starting from 1, corresponding to different color categories)
    pred_labels = torch.argmax(logits, dim=1) + 1  
    
    # -------------------------- 2. Upsample prediction labels to full point cloud (preserve original logic) --------------------------
    xyz_full = xyz_full.squeeze()  # Remove redundant dimensions
    chunk_len = xyz_full.shape[0] // N_CHUNKS + 1
    closest_idx_list = []
    
    # Chunked nearest-neighbor computation to avoid GPU memory overflow
    for i in range(N_CHUNKS):
        cur_chunk = xyz_full[chunk_len*i:chunk_len*(i+1)].cuda()
        # Compute distance between current chunk and downsampled point cloud
        dist = torch.norm(xyz_sub.unsqueeze(0) - cur_chunk.unsqueeze(1), dim=-1)
        # Find nearest neighbor indices
        min_idxs = torch.min(dist, 1)[1]
        closest_idx_list.append(min_idxs)
    
    # Merge all chunk indices and map to full point cloud prediction labels
    all_nn_idxs = torch.cat(closest_idx_list, axis=0)
    pred_full = pred_labels[all_nn_idxs].cpu().numpy()  # [N_full,] convert to numpy array
    
    # -------------------------- 3. Define fixed color mapping --------------------------
    # Can extend/adjust colors by category count; key=prediction label, value=(R,G,B) 0-255
    color_map = {
        1: (255, 0, 0),      # red
        2: (255, 255, 0),    # yellow
        3: (0, 0, 255),      # blue
        4: (0, 255, 0),      # green
        5: (0, 255, 255),    # cyan
        6: (255, 0, 255),    # magenta/purple
        7: (128, 128, 128),  # gray
        8: (255, 165, 0),    # orange
        9: (139, 69, 19),    # brown
        10: (240, 230, 140)  # khaki
    }
    # Default color for unmatched labels (black)
    default_color = (0, 0, 0)
    
    # -------------------------- 4. Process point cloud coordinates and colors --------------------------
    # Convert full point cloud coordinates to numpy array
    xyz_np = xyz_full.cpu().numpy()  # [N_full, 3]
    num_points = xyz_np.shape[0]
    
    # Assign color to each point
    colors = []
    for label in pred_full:
        colors.append(color_map.get(int(label), default_color))
    colors = np.array(colors, dtype=np.uint8)  # [N_full, 3]
    
    # -------------------------- 5. Save as PLY file --------------------------
    # Write PLY header (ASCII format)
    with open(save_path, 'w') as f:
        # PLY file header definition
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write(f'element vertex {num_points}\n')
        f.write('property float x\n')
        f.write('property float y\n')
        f.write('property float z\n')
        f.write('property uchar red\n')
        f.write('property uchar green\n')
        f.write('property uchar blue\n')
        f.write('end_header\n')
        
        # Write point coordinates and colors line by line
        for i in range(num_points):
            x, y, z = xyz_np[i]
            r, g, b = colors[i]
            f.write(f'{x:.6f} {y:.6f} {z:.6f} {r} {g} {b}\n')
    
    print(f"Prediction saved as PLY file: {save_path}")
    return save_path

# Single-sample data loading function (reuses Eval3dcom core logic)
def load_single_data(data_path, category, textembeds="clip", decorated=True):
    """Load single 3DCompat sample (fix device mismatch)."""
    # 1. Unify device (prefer CUDA, fallback to CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return_dict = {}
    
    # 2. Load core data (remove .cpu(), move to target device)
    mask_pts = torch.load(f"{data_path}/mask2points.pt", map_location=device)
    pts_xyz = torch.load(f"{data_path}/points.pt", map_location=device)
    normal = torch.load(f"{data_path}/normals.pt", map_location=device)
    pts_rgb = torch.load(f"{data_path}/rgb.pt", map_location=device) * 255
    # # debug: remove color
    # pts_rgb = torch.zeros_like(pts_rgb)
    gt = get_part_labels(pts_xyz, mask_pts) + 1  # 0 reserved for unknown
    gt = gt.to(device)  # Ensure gt on target device
    
    # 3. Text encoding (ensure text model and inputs on target device)
    with open(f"{data_path}/mask_labels.txt", "r") as f:
        labels = f.read().splitlines()
    if decorated:
        labels = [f"{part} of a {category}" for part in labels]
    
    # Initialize text model (move to target device)
    if textembeds == 'mpnet':
        model_name = "sentence-transformers/all-mpnet-base-v2"
    else:
        model_name = "google/siglip-base-patch16-224"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    text_model = AutoModel.from_pretrained(model_name).eval().to(device)  # Move model to device
    
    with torch.no_grad():
        # Move text input tensors to target device
        inputs = tokenizer(
            labels, 
            padding="max_length", 
            truncation=True, 
            return_tensors="pt"
        ).to(device)  # Key: move inputs to device
        
        if textembeds == 'mpnet':
            outputs = text_model(**inputs)
            token_embeddings = outputs[0]
            input_mask = inputs["attention_mask"].unsqueeze(-1).expand(token_embeddings.size())
            text_feat = torch.sum(token_embeddings * input_mask, 1) / torch.clamp(input_mask.sum(1), min=1e-9)
        else:
            text_feat = text_model.get_text_features(**inputs)
    
    # Normalize and ensure on target device
    text_feat = text_feat / (text_feat.norm(dim=-1, keepdim=True) + 1e-12)
    text_feat = text_feat.to(device)

    # 4. Data preparation (ensure returned tensors on target device)
    return_dict = prep_points_val3d(pts_xyz.cpu().numpy(), pts_rgb.cpu().numpy(), normal.cpu().numpy(), gt.cpu().numpy(), pts_xyz.cpu().numpy(), gt.cpu().numpy())

    # Move all to device
    for k in return_dict:
        if isinstance(return_dict[k], torch.Tensor):
            return_dict[k] = return_dict[k].to(device)
    
    # 5. Fill return dict (all tensors on same device)
    return_dict['label_embeds'] = text_feat.to(device)
    return_dict['class_name'] = category
    return_dict["xyz_visualization"] = pts_xyz.float().to(device)  # Reuse pts_xyz already on device
    return_dict["offset"] = return_dict["offset"].to(device)  # Ensure offset on device
    
    # Optional: explicitly use cuda when GPU is available
    # for k, v in return_dict.items():
    #     if isinstance(v, torch.Tensor):
    #         return_dict[k] = v.cuda()
    
    return return_dict

# Main entry
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Single 3DCompat sample evaluation")
    parser.add_argument("--data_path", 
                        type=str, 
                        default="data_test/coarse_b29_0cb_sample", 
                        help="Path to single sample (e.g. xxx/coarse_0001)")
    parser.add_argument("--category", 
                        type=str, 
                        default="vast", 
                        help="Data category (e.g. chair, table)")
    parser.add_argument("--checkpoint_path", 
                        type=str, 
                        default="dataset/checkpoints/ours_final.pth", 
                        help="Model checkpoint path")
    parser.add_argument("--save_path", 
                        type=str, 
                        default="results/sampling.ply",)
    parser.add_argument("--net_type", 
                        type=str, 
                        default="net8", 
                        help="Network type (net1, net8, etc.)")
    parser.add_argument("--textembeds", 
                        type=str, 
                        default="clip", 
                        help="Text embedding type (clip/mpnet)")
    args = parser.parse_args()


    # 1. Load model
    torch.manual_seed(123)
    if args.net_type in ['net1', 'net2']:
        model = load_model(args.checkpoint_path)
    else:
        model = PointSemSegWithDecoder(args=args)
        model.load_state_dict(torch.load(args.checkpoint_path)["model_state_dict"], strict=True)
    
    model = model.eval().cuda()
    temperature = np.exp(model.ln_logit_scale.item()) if hasattr(model, 'ln_logit_scale') else 1.0

    # 2. Load single sample
    data = load_single_data(args.data_path, args.category, args.textembeds)

    # 3. Model inference
    with torch.no_grad():
        data['mask_offset'] = torch.tensor([data['label_embeds'].shape[0]], device="cuda")
        model_output = model(data)
        
        # Extract model output
        if isinstance(model_output, (tuple, list)):
            net_out = model_output[0]
        elif isinstance(model_output, torch.Tensor):
            net_out = model_output
        else:
            raise TypeError(f"Unsupported model output format: {type(model_output)}")

    # 4. Compute 3D IoU and save
    save_path = compute_3d_iou_single(
        net_out=net_out,
        text_embeds=data['label_embeds'],
        temperature=temperature,
        cat=data['class_name'],
        xyz_sub=data['coord'],
        xyz_full=data['xyz_full'],
        gt_full=data['gt_full'],
        save_path=args.save_path
    )

    # 5. Print results
    print("Single-sample evaluation result:")
    print(f"Data path: {args.data_path}")
    print(f"Category: {args.category}")
    print(f"Save path: {save_path}")
