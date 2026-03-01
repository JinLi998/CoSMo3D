# evaluate and visualize
import torch
import matplotlib.pyplot as plt
import os
import torch.nn.functional as F
import numpy as np
import open3d as o3d
from sklearn.neighbors import NearestNeighbors

from .utils import preprocess_pcd, process_embeddings, SimilarityAnalyzer
from common.utils import visualize_pt_labels, visualize_pt_heatmap


def batch_iou(mask1, mask2): # both mask1 and mask2 are binary, batched and flattened
    # of shape (BS, H*W)
    union_binary = ((mask1 + mask2)>0)*1 # cur_view_n_masks, (H*W)
    union_area = union_binary.sum(dim=1)
    intersection_binary = mask1 * mask2 # cur_view_n_masks, (H*W)
    intersection_area = intersection_binary.sum(dim=1)
    iou = intersection_area / (union_area+1e-12)
    return iou




import numpy as np
import torch
import open3d as o3d
from sklearn.neighbors import NearestNeighbors
import torch.nn.functional as F

import numpy as np
import torch
import open3d as o3d
from sklearn.neighbors import NearestNeighbors
import torch.nn.functional as F

def compute_3d_iou_upsample_pro(
        pred,  # Downsampled point embeddings [n_subsampled_pts, feat_dim]
        part_text_embeds,  # Text embeddings [n_semantics, feat_dim]
        temperature,
        cat,
        xyz_sub,  # Downsampled point coordinates
        xyz_full,  # Full point coordinates [n_pts, 3]
        gt_full,  # Ground truth labels for full points
        panoptic=False,
        N_CHUNKS=1,
        visualize_seg=False,
        xyz_visualization=None,
        savepath=None,
        save_subsampled_seg=True
    ):
    xyz_full = xyz_full.squeeze()
    device = pred.device
    n_semantics = part_text_embeds.shape[0]  # Number of semantic categories
    # n_clusters = max(2 * n_semantics, 8)  # Cluster count = 2x semantics
    if n_semantics<3:
        n_clusters = 8
    else:
        n_clusters = int(n_semantics*2)


    # 1. Cluster to get initial parts
    seg_labels_sub, info = process_embeddings(pred, part_text_embeds, n_clusters)
    cluster_labels = torch.from_numpy(info["cluster_labels"]).to(device)
    
    # Ensure we get all unique cluster IDs (convert to Python int for dict keys)
    unique_clusters = [int(cid) for cid in torch.unique(cluster_labels).cpu().numpy()]

    # 2. Compute similarity between each cluster and each semantic category
    part_masks = {cid: (cluster_labels == cid) for cid in unique_clusters}

    cluster_feats = {}
    for cid in unique_clusters:
        mask = part_masks[cid]
        cluster_feats[cid] = torch.mean(pred[mask], dim=0, keepdim=True) if mask.sum() > 0 else torch.zeros(1, pred.shape[1], device=device)

    # 3. Analyze best part combination for each semantic category
    semantic_clusters = []
    for sem_id in range(n_semantics):
        sem_embed = part_text_embeds[sem_id:sem_id+1]
        cluster_sims = [(cid, F.cosine_similarity(feat, sem_embed, dim=1).item()) 
                       for cid, feat in cluster_feats.items()]
        sorted_clusters = [cid for cid, _ in sorted(cluster_sims, key=lambda x: x[1], reverse=True)]

        # Analyze merge process
        merged_groups, group_sims = [], []
        current_group = []
        for cid in sorted_clusters:
            current_group.append(cid)
            if not all(c in part_masks for c in current_group):
                continue
            mask = torch.any(torch.stack([part_masks[lid] for lid in current_group]), dim=0)
            feat = torch.mean(pred[mask], dim=0, keepdim=True) if mask.sum() > 0 else torch.zeros_like(sem_embed)
            group_sims.append(F.cosine_similarity(feat, sem_embed, dim=1).item())
            merged_groups.append(current_group.copy())

        # Determine best combination
        best_idx = np.argmax(group_sims) if group_sims else 0
        semantic_clusters.append(merged_groups[best_idx] if merged_groups else [])
        # print(f"语义类别 {sem_id+1} 最佳组合: {semantic_clusters[-1]}, 相似度: {group_sims[best_idx]:.4f}" if group_sims else f"语义类别 {sem_id+1} 无有效组合")

    # 4. Generate multi-semantic labels (1..n_semantics, 0 for background)
    semantic_sub = torch.zeros_like(cluster_labels, dtype=int)
    for sem_id in range(n_semantics):
        for cid in semantic_clusters[sem_id]:
            if cid in part_masks:
                # Avoid label conflict
                if semantic_sub[cluster_labels == cid].sum() == 0:
                    semantic_sub[cluster_labels == cid] = sem_id + 1



    # 6. Upsample to full point cloud - fix NearestNeighbors param error
    xyz_sub_np = xyz_sub.cpu().numpy()
    xyz_full_np = xyz_full.cpu().numpy()
    semantic_sub_np = semantic_sub.cpu().numpy()
    
    all_indices = []
    chunk_len = xyz_full.shape[0] // N_CHUNKS + 1
    # Initialize NearestNeighbors model (correct params)
    nn_model = NearestNeighbors(n_neighbors=1, algorithm='kd_tree')
    nn_model.fit(xyz_sub_np)  # Fit model first
    
    for i in range(N_CHUNKS):
        chunk = xyz_full_np[chunk_len*i : chunk_len*(i+1)]
        # Query using fitted model
        _, indices = nn_model.kneighbors(chunk)
        all_indices.append(indices.squeeze())
    
    pred_full = torch.from_numpy(semantic_sub_np[np.concatenate(all_indices)]).cpu()

    # 7. Compute multi-category evaluation metrics
    pred_np, gt_np = pred_full.numpy(), gt_full.squeeze().numpy()
    acc = (pred_np == gt_np).mean()
    
    ious = []
    for sem_id in range(1, n_semantics+1):
        I = np.logical_and(pred_np == sem_id, gt_np == sem_id).sum()
        U = np.logical_or(pred_np == sem_id, gt_np == sem_id).sum()
        ious.append(I / U if U > 0 else 0.0)
    miou = np.mean(ious)

    # 8. Save final results
    if savepath:
        visualize_pt_labels(xyz_visualization.cpu(), pred_full, save_path=savepath,
                          save_rendered_path=savepath.replace('.ply', '.png'))
        visualize_pt_labels(xyz_visualization.cpu(), gt_full, 
                          save_path=savepath.replace('.ply', '_gt.ply'),
                          save_rendered_path=savepath.replace('.ply', '_gt.png'))

    return miou, acc

from sklearn.decomposition import PCA
def save_feats_to_ply(features, xyz, save_path, normalize_feats=True):
    """
    Fix param order: features first, then coordinates, to match call convention.
    Use PCA to reduce high-dim features to 3D and map to RGB colors, save point cloud as PLY.
    
    Args:
        features: Point cloud features, shape [N, C] (e.g. [2179, 768])
        xyz: Point cloud coordinates, shape [N, 3] (e.g. [2179, 3])
        save_path: Path to save PLY file
        normalize_feats: Whether to normalize features (recommend True)
    """
    # Convert to numpy array (supports PyTorch tensor)
    if isinstance(features, torch.Tensor):
        features = features.detach().cpu().numpy()
    if isinstance(xyz, torch.Tensor):
        xyz = xyz.detach().cpu().numpy()
    
    # Validate input shapes (core fix: ensure coordinates are [N, 3])
    N = features.shape[0]
    assert xyz.shape == (N, 3), f"Coordinate shape must be [N, 3], got {xyz.shape} (N={N})"
    assert len(features.shape) == 2, f"Features must be 2D array, got {features.shape}"
    
    # Normalize features to [0, 1]
    if normalize_feats:
        feats_min = features.min(axis=0, keepdims=True)
        feats_max = features.max(axis=0, keepdims=True)
        feats_range = feats_max - feats_min
        feats_range[feats_range < 1e-8] = 1e-8  # Avoid div by zero
        features = (features - feats_min) / feats_range
    
    # PCA reduce to 3D (preserve main feature variance)
    pca = PCA(n_components=3)
    feats_3d = pca.fit_transform(features)  # [N, 3]
    
    # Map to RGB color range [0, 255]
    pca_min, pca_max = feats_3d.min(axis=0), feats_3d.max(axis=0)
    pca_range = pca_max - pca_min
    pca_range[pca_range < 1e-8] = 1e-8
    rgb = ((feats_3d - pca_min) / pca_range * 255).astype(np.uint8)  # [N, 3]
    
    # Write PLY file
    with open(save_path, 'w') as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {N}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for i in range(N):
            x, y, z = xyz[i]
            r, g, b = rgb[i]
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b}\n")
   
def compute_3d_iou_upsample(
        pred, # n_subsampled_pts, feat_dim
        part_text_embeds, # n_parts, feat_dim
        temperature,
        cat,
        xyz_sub,
        xyz_full, # n_pts, 3
        gt_full, # n_pts,
        panoptic = False,
        N_CHUNKS=1,
        visualize_seg=False,
        visualize_all_heatmap=False,
        xyz_visualization=None,
        savepath=None
        ):
    xyz_full = xyz_full.squeeze()
    # first get each point's logits
    logits = pred @ part_text_embeds.T # n_pts, n_mask

    
    if panoptic:
        pred_softmax = torch.nn.Softmax(dim=1)(logits * temperature)
    else:
        # prepend 0 as no label since if all queries are negative we should report no label
        logits_prepend0 = torch.cat([torch.zeros(logits.shape[0],1).cuda(), logits],axis=1)
        pred_softmax = torch.nn.Softmax(dim=1)(logits_prepend0 * temperature)
    
    
    # assign to nearest neighbor
    chunk_len = xyz_full.shape[0]//N_CHUNKS+1
    closest_idx_list = []
    for i in range(N_CHUNKS):
        cur_chunk = xyz_full[chunk_len*i:chunk_len*(i+1)]
        dist_all = (xyz_sub.unsqueeze(0) - cur_chunk.cuda().unsqueeze(1))**2 # 300k,5k,3
        cur_dist = (dist_all.sum(dim=-1))**0.5 # 300k,5k
        min_idxs = torch.min(cur_dist, 1)[1]
        del cur_dist
        closest_idx_list.append(min_idxs)
    all_nn_idxs = torch.cat(closest_idx_list,axis=0)
    all_probs = pred_softmax[all_nn_idxs]
    
    # now argmax
    if panoptic:
        pred_full = all_probs.argmax(dim=1).cpu() + 1# here, no unlabeled, 1,...n_part correspond to actual part assignment
    else:
        pred_full = all_probs.argmax(dim=1).cpu()# here, 0 is unlabeled, 1,...n_part correspond to actual part assignment
    
    acc = ((pred_full == gt_full)*1).sum() / pred_full.shape[0]
    pred_np = pred_full.numpy()
    label_np = gt_full.squeeze().numpy()

    # '''if visualize_seg:
    #     # visualize on original scale xyz
    #     visualize_pt_labels(xyz_visualization.cpu(), gt_full.squeeze().cpu(), save_path=f"{cat}_gt")
    #     visualize_pt_labels(xyz_visualization.cpu(), pred_full.cpu(), save_path=f"{cat}_pred")

    # if visualize_all_heatmap:
    #     xyz_full_ori_axis = torch.cat([-xyz_full[:,0].reshape(-1,1), xyz_full[:,2].reshape(-1,1), xyz_full[:,1].reshape(-1,1)], dim=1)
    #     for i in range(all_probs.shape[1]-1): # all queries
    #         cur_scores = all_probs[:,i+1] # skip 0 which corresponds to unlabeled
    #         visualize_pt_heatmap(xyz_full_ori_axis.cpu(), cur_scores.cpu(), save_path=f"{cat}_heatmap{i}")'''

    # Save segmentation results
    if savepath is not None:
            # Check if savepath ends with .ply
            assert savepath.endswith('.ply')

            pngsavepath = savepath.replace('.ply', '.png')

            # Save point cloud or rendered results
            # print('debug : ', xyz_full_ori_axis.shape, pred_full.shape)
            caption_list = visualize_pt_labels(xyz_visualization.cpu(), pred_full.cpu(), save_path=savepath, save_rendered_path=pngsavepath)
            # Replace .ply with _gt.ply
            savepath = savepath.replace(".ply", "_gt.ply") 
            pngsavepath = pngsavepath.replace(".png", "_gt.png")
            caption_list = visualize_pt_labels(xyz_visualization.cpu(), gt_full.squeeze().cpu(), save_path=savepath, save_rendered_path=pngsavepath)
            print('caption_list:', caption_list)
            # Convert features to colors and save
            featssavepath = savepath.replace("_gt.ply", "_feats.ply")
            save_feats_to_ply(pred, xyz_sub, featssavepath)
          
    # get full iou
    part_ious = []
    for part in range(part_text_embeds.shape[0]):
        I = np.sum(np.logical_and(pred_np == part+1, label_np == part+1))
        U = np.sum(np.logical_or(pred_np == part+1, label_np == part+1))
        if U == 0:
            pass
        else:
            iou = I / float(U)
            part_ious.append(iou)
    full_miou = np.mean(part_ious)
    full_macc = acc.item()
    return full_miou, full_macc


def visualize_3d_upsample(
        pred, # n_subsampled_pts, feat_dim
        part_text_embeds, # n_parts, feat_dim
        temperature,
        xyz_sub,
        xyz_full, # n_pts, 3
        panoptic = False,
        N_CHUNKS=1,
        heatmap = False, # heatmap or segmentation
        savepath = None, # whether to save the rendered point cloud or not
        ):
    xyz_full = xyz_full.squeeze()
    logits = pred @ part_text_embeds.T # n_pts, n_mask

    if panoptic:
        pred_softmax = torch.nn.Softmax(dim=1)(logits * temperature)
    else:
        # prepend 0 as no label
        logits_prepend0 = torch.cat([torch.zeros(logits.shape[0],1).cuda(), logits],axis=1)
        pred_softmax = torch.nn.Softmax(dim=1)(logits_prepend0 * temperature)
    
    chunk_len = xyz_full.shape[0]//N_CHUNKS+1
    closest_idx_list = []
    for i in range(N_CHUNKS):
        cur_chunk = xyz_full[chunk_len*i:chunk_len*(i+1)]
        dist_all = (xyz_sub.unsqueeze(0) - cur_chunk.cuda().unsqueeze(1))**2 # 300k,5k,3
        cur_dist = (dist_all.sum(dim=-1))**0.5 # 300k,5k
        min_idxs = torch.min(cur_dist, 1)[1]
        del cur_dist
        closest_idx_list.append(min_idxs)
    all_nn_idxs = torch.cat(closest_idx_list,axis=0)
    # just inversely weight all points
    all_probs = pred_softmax[all_nn_idxs]
    all_logits = logits[all_nn_idxs]
    
    # now argmax
    if panoptic:
        pred_full = all_probs.argmax(dim=1).cpu() + 1# here, no unlabeled, 1,...n_part correspond to actual part assignment
    else:
        pred_full = all_probs.argmax(dim=1).cpu()# here, 0 is unlabeled, 1,...n_part correspond to actual part assignment
    
    # convert back to the original coordinate system for visualization
        xyz_full_ori_axis = torch.cat([-xyz_full[:,0].reshape(-1,1), xyz_full[:,2].reshape(-1,1), xyz_full[:,1].reshape(-1,1)], dim=1)
    if heatmap:
        for i in range(all_probs.shape[1]-1): # all queries
            cur_scores = all_logits[:,i]
            visualize_pt_heatmap(xyz_full_ori_axis.cpu(), cur_scores.cpu())
    else: # segmentation
        if savepath is not None:
            # Check if savepath ends with .ply
            assert savepath.endswith('.ply')

            pngsavepath = savepath.replace('.ply', '.png')

            # Save point cloud or rendered results
            # print('debug : ', xyz_full_ori_axis.shape, pred_full.shape)
            caption_list = visualize_pt_labels(xyz_full_ori_axis.cpu(), pred_full.cpu(), save_path=savepath, save_rendered_path=pngsavepath)
        else:
            caption_list = visualize_pt_labels(xyz_full_ori_axis.cpu(), pred_full.cpu())

        

    return caption_list

def visualize_3d_upsample_render(
        pred, # n_subsampled_pts, feat_dim
        part_text_embeds, # n_parts, feat_dim
        temperature,
        xyz_sub,
        xyz_full, # n_pts, 3
        xyz_visualization,
        img_save_path,
        panoptic = False,
        N_CHUNKS=1,
        heatmap = False # heatmap or segmentation
        ):
    xyz_full = xyz_full.squeeze()
    logits = pred @ part_text_embeds.T # n_pts, n_mask

    if panoptic:
        pred_softmax = torch.nn.Softmax(dim=1)(logits * temperature)
    else:
        # prepend 0 as no label
        logits_prepend0 = torch.cat([torch.zeros(logits.shape[0],1).cuda(), logits],axis=1)
        pred_softmax = torch.nn.Softmax(dim=1)(logits_prepend0 * temperature)

    chunk_len = xyz_full.shape[0]//N_CHUNKS+1
    closest_idx_list = []
    for i in range(N_CHUNKS):
        cur_chunk = xyz_full[chunk_len*i:chunk_len*(i+1)]
        dist_all = (xyz_sub.unsqueeze(0) - cur_chunk.cuda().unsqueeze(1))**2 # 300k,5k,3
        cur_dist = (dist_all.sum(dim=-1))**0.5 # 300k,5k
        min_idxs = torch.min(cur_dist, 1)[1]
        del cur_dist
        closest_idx_list.append(min_idxs)
    all_nn_idxs = torch.cat(closest_idx_list,axis=0)
    # just inversely weight all points
    all_probs = pred_softmax[all_nn_idxs]
    
    # now argmax
    if panoptic:
        pred_full = all_probs.argmax(dim=1).cpu() + 1# here, no unlabeled, 1,...n_part correspond to actual part assignment
    else:
        pred_full = all_probs.argmax(dim=1).cpu()# here, 0 is unlabeled, 1,...n_part correspond to actual part assignment
    
    # convert back to the original coordinate system for visualization
    visualize_pt_labels(xyz_visualization.cpu(), pred_full.cpu(), save_rendered_path=img_save_path)
    return

# note the two methods below are heuristic---they are not used in any formal
# evaluation, they are only for visualization/diagnostics during training
def compute_overall_iou_objwise(pred, # n_pts, feat_dim
                                text_embeds, # n_mask, feat_dim
                                masks, # n_masks, h, w - binary in 2d
                                mask_view_idxs, # n_masks, each has a view index, -1 for padding
                                point2face, # n_pts
                                pixel2face, # 10,H,W
                                temperature
                                ):
    # the text embedding is normalized to norm 1
    # the pred is not normalized since it's whatever the model outputs
    # we regard anything > 0 as a match and <= as not a match when obtaining masks
    # we can further adjust this if we decide to normalize the pred here (not during training since then the 
    # contrastive loss will be affected)
    THRESHOLD = 0.6
    # first get each point's logits
    n_views, H, W = pixel2face.shape
    masks_flattened = masks.view(masks.shape[0],-1) # n_masks, (H*W)

    # for each view, get a H,W,n_mask distribution
    all_ious = []
    for i in range(n_views):
        if torch.sum((mask_view_idxs==i)*1) == 0:
            continue
        relevant_masks = masks_flattened[mask_view_idxs==i,:]# cur_view_n_masks,(H*W)
        relevant_text_embeds = text_embeds[mask_view_idxs==i,:]# cur_view_n_masks,feat_dim
        logits = pred @ relevant_text_embeds.T # n_pts, cur_view_n_masks

        # get binary mask of (H*W)*5000
        cur_faces = pixel2face[i,:,:].view(-1,1)
        pixel2point_mask = (cur_faces == point2face.view(1,-1))*1.0 # (H*W),n_pts
        # this is binary where all points contributing to each pixel is 1
        # need to normalize
        n_pts = pixel2point_mask.sum(dim=1)
        normalized_mask = pixel2point_mask / (n_pts+1e-12).view(-1,1) # (H*W),n_pts
        # should be
        # [1/3 0 0 ... 1/3 ... 1/3]
        # [0  1/2 1/2 ...0  .....0]
        view_logits = normalized_mask @ logits # (H*W),cur_view_n_masks
        # append 0 - in case only one mask available
        view_logits_append0 = torch.cat([view_logits, torch.zeros(view_logits.shape[0],1).cuda()],axis=1)
        # view_logits: for each pixel, we get an average of the logits of all points that correspond to this pixel 
        view_softmax = torch.nn.Softmax(dim=1)(view_logits_append0 * temperature)[:,:-1]
        
        # we need a threshold and can't just take max! because the majority of pixels should not correspond to any mask
        # to avoid setting a manual threshold, we use a coefficient * max across all labels
        # the premise is that since we provide ground truth text queries, the max should be informative (large, close to 1)
        # and by taking a coefficient e.g. 0.5 we are taking some relatively high-correspondence regions
        thres_binary_mask = (view_softmax > THRESHOLD)*1 # (H*W),cur_view_n_masks

        # get threshold IoU
        iou = batch_iou(thres_binary_mask.T, relevant_masks)

        all_ious.append(iou)

    all_iou_vec = torch.cat(all_ious)
    mean_iou = all_iou_vec.mean().item()
    return mean_iou

# visualize predicted masks
def viz_pred_mask(pred, # n_pts, feat_dim
                  text_embeds, # n_mask, feat_dim
                  texts, # list of n_mask
                  masks, # n_masks, h, w - binary in 2d
                  mask_view_idxs, # n_masks, each has a view index, -1 for padding
                  point2face, # 5000
                  pixel2face, # 10,H,W
                  n_epoch, # which epoch we are evaluating
                  obj_visualize_idx, # which object we are evaluating
                  prefix, # prefix for saving
                  temperature,
                  threshold=0.6
                  ):
    i_vals = [2,3,6]
    j_vals = [0,1]
    for i in i_vals:
        for j in j_vals:
            # first get relevant masks and texts
            if torch.sum((mask_view_idxs==i)*1) <= j:
                return
            relevant_masks = masks[mask_view_idxs==i,:,:][j,:,:] # h,w
            H,W = relevant_masks.shape
            relevant_text_embeds = text_embeds[mask_view_idxs==i,:] # n_masks, feat_dim
            relevant_text = [i for (i, v) in zip(texts, (mask_view_idxs==i).tolist()) if v][j][0]
            # first get each point's logits
            logits = pred @ relevant_text_embeds.T # n_pts, n_curview_masks

            # first get binary mask of (H*W)*5000
            cur_faces = pixel2face[i,:,:].view(-1,1)
            pixel2point_mask = (cur_faces == point2face.view(1,-1))*1.0 # (H*W),n_pts
            # this is binary where all points contributing to each pixel is 1
            # need to normalize
            n_pts = pixel2point_mask.sum(dim=1)
            normalized_mask = pixel2point_mask / (n_pts+1e-12).view(-1,1) # (H*W),n_pts
            # should be
            # [1/3 0 0 ... 1/3 ... 1/3]
            # [0  1/2 1/2 ...0  .....0]
            view_logits = normalized_mask @ logits # (H*W),cur_n_masks
            # we append a new 0 category just in case in current view there is only one mask, in which case we would have gotten all 1 after softmax otherwise
            view_logits_append0 = torch.cat([view_logits, torch.zeros(view_logits.shape[0],1).cuda()],axis=1)
            # view_logits: for each pixel, we get an average of the logits of all points that correspond to this pixel 
            view_softmax = torch.nn.Softmax(dim=1)(view_logits_append0 * temperature)
            view_softmax_heatmap = view_softmax[:,j].view(H,W)
            os.makedirs(f"training_checkpts/visualization/{prefix}_obj{obj_visualize_idx}/", exist_ok=True)

            # visualize heatmap and gt
            plt.clf()
            plt.imshow(view_softmax_heatmap.cpu())
            plt.colorbar()
            plt.title(relevant_text)
            plt.savefig(f"training_checkpts/visualization/{prefix}_obj{obj_visualize_idx}/view{i}mask{j}_{n_epoch}_pred_heatmap.png")

            plt.clf()
            plt.imshow(((view_softmax_heatmap>threshold)*1).cpu())
            plt.title(relevant_text)
            plt.savefig(f"training_checkpts/visualization/{prefix}_obj{obj_visualize_idx}/view{i}mask{j}_{n_epoch}_pred_mask.png")

            plt.clf()
            plt.imshow(relevant_masks.cpu())
            plt.title(relevant_text)
            plt.savefig(f"training_checkpts/visualization/{prefix}_obj{obj_visualize_idx}/view{i}mask{j}_gt_heatmap.png")
        
    return


def get_feature(model, xyz, rgb, normal): # evaluate loader can only have batch size=1
    data = preprocess_pcd(xyz.cuda(), rgb.cuda(), normal.cuda())
    with torch.no_grad():
        for key in data.keys():
            if isinstance(data[key], torch.Tensor) and "full" not in key:
                data[key] = data[key].cuda(non_blocking=True)
        net_out = model(x=data) # n_pts,dim_feats
        xyz_sub = data["coord"] # n_pts,3
    return xyz_sub, net_out
