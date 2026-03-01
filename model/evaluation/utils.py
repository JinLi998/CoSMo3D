import torch
from types import SimpleNamespace
from model.backbone.pt3.model import PointSemSeg
import numpy as np
import random
from transformers import AutoTokenizer, AutoModel
import open3d as o3d
from common.utils import visualize_pts




def load_model(checkpoint_path):
    args = SimpleNamespace()
    model = PointSemSeg(args=args, dim_output=768)
    model.load_state_dict(torch.load(checkpoint_path)["model_state_dict"])
    model.eval()
    model = model.cuda()
    return model

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def fnv_hash_vec(arr):
    """
    FNV64-1A
    """
    assert arr.ndim == 2
    # Floor first for negative coordinates
    arr = arr.copy()
    arr = arr.astype(np.uint64, copy=False)
    hashed_arr = np.uint64(14695981039346656037) * np.ones(
        arr.shape[0], dtype=np.uint64
    )
    for j in range(arr.shape[1]):
        hashed_arr *= np.uint64(1099511628211)
        hashed_arr = np.bitwise_xor(hashed_arr, arr[:, j])
    return hashed_arr


def grid_sample_numpy(xyz, rgb, normal, grid_size): # this should hopefully be 5000 or close
    xyz = xyz.cpu().numpy()
    rgb = rgb.cpu().numpy()
    normal = normal.cpu().numpy()

    scaled_coord = xyz / np.array(grid_size)
    grid_coord = np.floor(scaled_coord).astype(int)
    min_coord = grid_coord.min(0)
    grid_coord -= min_coord
    scaled_coord -= min_coord
    min_coord = min_coord * np.array(grid_size)
    key = fnv_hash_vec(grid_coord)
    idx_sort = np.argsort(key)
    key_sort = key[idx_sort]
    _, inverse, count = np.unique(key_sort, return_inverse=True, return_counts=True)
    idx_select = (
        np.cumsum(np.insert(count, 0, 0)[0:-1])
        + np.random.randint(0, count.max(), count.size) % count
    )
    idx_unique = idx_sort[idx_select]

    grid_coord = grid_coord[idx_unique]
    
    xyz = torch.tensor(xyz[idx_unique]).cuda()
    rgb = torch.tensor(rgb[idx_unique]).cuda()
    normal = torch.tensor(normal[idx_unique]).cuda()
    grid_coord = torch.tensor(grid_coord).cuda()

    return xyz, rgb, normal, grid_coord
    

def preprocess_pcd(xyz, rgb, normal): # rgb should be 0-1
    assert rgb.max() <=1
    # normalize
    # this is the same preprocessing I do before training
    center = xyz.mean(0)
    scale = max((xyz - center).abs().max(0)[0])
    xyz -= center
    xyz *= (0.75 / float(scale)) # put in 0.75-size box

    # axis swap
    xyz = torch.cat([-xyz[:,0].reshape(-1,1), xyz[:,2].reshape(-1,1), xyz[:,1].reshape(-1,1)], dim=1)

    # center shift
    xyz_min = xyz.min(dim=0)[0]
    xyz_max = xyz.max(dim=0)[0]
    xyz_max[2] = 0
    shift = (xyz_min+xyz_max)/2
    xyz -= shift

    # subsample/upsample to 5000 pts for grid sampling
    if xyz.shape[0] != 5000:
        random_indices = torch.randint(0, xyz.shape[0], (5000,))
        pts_xyz_subsampled = xyz[random_indices]
        pts_rgb_subsampled = rgb[random_indices]
        normal_subsampled = normal[random_indices]
    else:
        pts_xyz_subsampled = xyz
        pts_rgb_subsampled = rgb
        normal_subsampled = normal

    # grid sampling
    pts_xyz_gridsampled, pts_rgb_gridsampled, normal_gridsampled, grid_coord = grid_sample_numpy(pts_xyz_subsampled, pts_rgb_subsampled, normal_subsampled, 0.02)

    # another center shift, z=false
    xyz_min = pts_xyz_gridsampled.min(dim=0)[0]
    xyz_min[2] = 0
    xyz_max = pts_xyz_gridsampled.max(dim=0)[0]
    xyz_max[2] = 0
    shift = (xyz_min+xyz_max)/2
    pts_xyz_gridsampled -= shift
    xyz -= shift

    # normalize color
    pts_rgb_gridsampled = pts_rgb_gridsampled / 0.5 - 1

    # combine color and normal as feat
    feat = torch.cat([pts_rgb_gridsampled, normal_gridsampled], dim=1)

    data_dict = {}
    data_dict["coord"] = pts_xyz_gridsampled
    data_dict["feat"] = feat
    data_dict["grid_coord"] = grid_coord
    data_dict["xyz_full"] = xyz
    data_dict["offset"] = torch.tensor([pts_xyz_gridsampled.shape[0]]).to(pts_xyz_gridsampled.device)
    return data_dict


def encode_text(texts):
    siglip = AutoModel.from_pretrained("google/siglip-base-patch16-224") # dim 768 #"google/siglip-so400m-patch14-384")
    tokenizer = AutoTokenizer.from_pretrained("google/siglip-base-patch16-224")#"google/siglip-so400m-patch14-384")
    inputs = tokenizer(texts, padding="max_length", return_tensors="pt")
    for key in inputs:
        inputs[key] = inputs[key].cuda()
    with torch.no_grad():
        text_feat = siglip.cuda().get_text_features(**inputs)
    text_feat = text_feat / (text_feat.norm(dim=-1, keepdim=True) + 1e-12)
    return text_feat

def read_ply(obj_path, visualize=True):
    pcd = o3d.io.read_point_cloud(obj_path)
    if visualize:
        visualize_pts(torch.tensor(np.asarray(pcd.points)), torch.tensor(np.asarray(pcd.colors)), save_path="actual")
    xyz = torch.tensor(np.asarray(pcd.points)).float()
    rgb = torch.tensor(np.asarray(pcd.colors)).float()
    normal = torch.tensor(np.asarray(pcd.normals)).float()
    return xyz, rgb, normal


def read_pcd(obj_path, visualize=True):
    pcd = o3d.io.read_point_cloud(obj_path)
    if visualize:
        visualize_pts(torch.tensor(np.asarray(pcd.points)), torch.tensor(np.asarray(pcd.colors)), save_path="actual")
    xyz = torch.tensor(np.asarray(pcd.points)).float()
    rgb = torch.tensor(np.asarray(pcd.colors)).float()
    normal = torch.tensor(np.asarray(pcd.normals)).float()
    return xyz, rgb, normal








####################
import os
import torch
import numpy as np
import open3d as o3d
from typing import List, Tuple, Dict, Optional
from sklearn.cluster import KMeans
import torch.nn.functional as F
import torch.nn as nn

def process_embeddings(shape_embeds: torch.Tensor, text_embeds: torch.Tensor, n_clusters: int) -> Tuple[torch.Tensor, dict]:
    """
    Process embedding vectors, perform clustering, compute similarities and generate category labels.
    
    Args:
        shape_embeds: Shape embedding vectors
        text_embeds: Text embedding vectors
        n_clusters: Number of clusters (hyperparameter)
    
    Returns:
        labels: Per-point category labels, starting from 1, 0 for unclassified
        info: Dict containing intermediate results, e.g. similarities, cluster centers
    """
    # Ensure inputs are proper tensors and move to CPU for processing
    if isinstance(shape_embeds, torch.Tensor):
        shape_embeds_np = shape_embeds.cpu().numpy()
    else:
        shape_embeds_np = np.array(shape_embeds)
    
    # 1. Run KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(shape_embeds_np)
    cluster_labels = kmeans.labels_  # Cluster labels start from 0
    
    # 2. Compute similarity between each cluster and text
    part_similarities = []
    for cluster_id in range(n_clusters):
        # Get all points in this cluster
        mask = (cluster_labels == cluster_id)
        if np.sum(mask) == 0:
            part_similarities.append((cluster_id, -1.0))  # Empty cluster: lowest similarity
            continue
        
        # Compute cluster mean embedding
        cluster_embeds = shape_embeds[mask]
        cluster_mean = torch.mean(cluster_embeds, dim=0, keepdim=True)
        
        # Compute cosine similarity with text embeddings
        sim = F.cosine_similarity(cluster_mean, text_embeds, dim=1).mean().item()
        part_similarities.append((cluster_id, sim))
    
    # 3. Sort clusters by similarity
    part_similarities.sort(key=lambda x: x[1], reverse=True)
    sorted_clusters = [cid for cid, _ in part_similarities]
    sorted_sims = [sim for _, sim in part_similarities]
    
    # 4. Determine best combination (here use all clusters)
    best_clusters = sorted_clusters  # Use all clusters
    
    # 5. Generate final labels (starting from 1, 0 for unselected)
    final_labels = np.zeros_like(cluster_labels, dtype=int)
    for idx, cluster_id in enumerate(best_clusters):
        final_labels[cluster_labels == cluster_id] = idx + 1  # Labels start from 1
    
    # Convert to PyTorch tensor
    final_labels_tensor = torch.from_numpy(final_labels).to(shape_embeds.device)
    
    # Organize return info
    info = {
        "cluster_labels": cluster_labels,
        "similarities": sorted_sims,
        "sorted_clusters": sorted_clusters,
        "best_clusters": best_clusters,
        "kmeans_model": kmeans
    }
    
    return final_labels_tensor, info



class SimilarityAnalyzer:
    """Part merge tool based on similarity change analysis"""
    
    def __init__(self, checkpoint_path: str, device: Optional[str] = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model(checkpoint_path)
    
    def _load_model(self, checkpoint_path: str):
        from model.evaluation.utils import load_model
        model = load_model(checkpoint_path)
        model.to(self.device)
        model.eval()
        return model
    
    @staticmethod
    def hsv_to_rgb(h: float, s: float, v: float) -> List[float]:
        if s == 0.0:
            return [v, v, v]
        i = int(h * 6.0)
        f = (h * 6.0) - i
        p = v * (1.0 - s)
        q = v * (1.0 - s * f)
        t = v * (1.0 - s * (1.0 - f))
        i = i % 6
        if i == 0:
            return [v, t, p]
        elif i == 1:
            return [q, v, p]
        elif i == 2:
            return [p, v, t]
        elif i == 3:
            return [p, q, v]
        elif i == 4:
            return [t, p, v]
        else:
            return [v, p, q]
    
    def cluster_parts(self, shape_embeds: torch.Tensor, min_clusters: int = 2, 
                     max_clusters: int = 20, selected_clusters: int = 8) -> torch.Tensor:
        cluster_labels = []
        for num_cluster in range(min_clusters, max_clusters):
            clustering = KMeans(n_clusters=num_cluster, random_state=0).fit(
                shape_embeds.cpu().numpy()
            )
            cluster_labels.append(clustering.labels_)
        
        cluster_labels = np.stack(cluster_labels, axis=0)
        cluster_labels = torch.from_numpy(cluster_labels).to(self.device)
        return cluster_labels[selected_clusters - 2]
    
    def cluster_parts_with_wrapper(self, shape_embeds: torch.Tensor, text_embeds: torch.Tensor, n_clusters: int) -> torch.Tensor:
        """Use wrapped function for clustering and label generation"""
        labels, _ = process_embeddings(shape_embeds, text_embeds, n_clusters)
        return labels
    
    def get_part_masks(self, seg_labels: torch.Tensor) -> Dict[int, torch.Tensor]:
        unique_labels = torch.unique(seg_labels)
        part_masks = {}
        
        for label in unique_labels:
            mask = (seg_labels == label)
            part_masks[label] = mask
            
        # Verify mask validity
        for label, mask in part_masks.items():
            print(f"Part {label} point count: {torch.sum(mask).item()}")
                
        print(f"Number of valid segmentation parts: {len(part_masks)}")
        return part_masks
    
    def compute_combined_similarity(self, part_labels: List[int], part_masks: Dict[int, torch.Tensor],
                                   shape_embeds: torch.Tensor, text_embeds: torch.Tensor) -> float:
        """Compute overall similarity between combined parts and text"""
        # Create combined mask
        combined_mask = torch.zeros_like(next(iter(part_masks.values())), dtype=torch.bool)
        for label in part_labels:
            combined_mask |= part_masks[label]
        
        # Compute combined features
        if torch.sum(combined_mask) == 0:
            return 0.0
            
        combined_feat = torch.mean(shape_embeds[combined_mask], dim=0, keepdim=True)
        cos_sim = F.cosine_similarity(combined_feat, text_embeds, dim=1)
        return cos_sim.mean().item()
    
    def sort_parts_by_similarity(self, part_masks: Dict[int, torch.Tensor], 
                                shape_embeds: torch.Tensor, text_embeds: torch.Tensor) -> List[int]:
        part_similarities = []
        for label, mask in part_masks.items():
            part_feat = torch.mean(shape_embeds[mask], dim=0, keepdim=True)
            sim = F.cosine_similarity(part_feat, text_embeds, dim=1).mean().item()
            part_similarities.append((label, sim))
        
        # Sort by similarity descending
        part_similarities.sort(key=lambda x: x[1], reverse=True)
        return [label for label, _ in part_similarities], [sim for _, sim in part_similarities]
    
    def analyze_similarity_changes(self, sorted_labels: List[int], part_masks: Dict[int, torch.Tensor],
                                  shape_embeds: torch.Tensor, text_embeds: torch.Tensor) -> Tuple[List[List[int]], List[float], List[float]]:
        """Analyze similarity changes during stepwise merge"""
        merged_groups = []
        group_similarities = []
        similarity_diffs = []  # Similarity change rate
        current_group = []
        
        for i, label in enumerate(sorted_labels):
            current_group.append(label)
            merged_groups.append(current_group.copy())
            
            # Compute overall similarity of current combination
            current_sim = self.compute_combined_similarity(current_group, part_masks, shape_embeds, text_embeds)
            group_similarities.append(current_sim)
            
            # Compute similarity change rate
            if i == 0:
                # First combination, no predecessor, change rate = 0
                similarity_diffs.append(0.0)
            else:
                # Compute similarity diff from previous combination
                diff = current_sim - group_similarities[i-1]
                # Compute relative change rate (divide by prev similarity to avoid scale issues)
                relative_diff = diff / abs(group_similarities[i-1]) if group_similarities[i-1] != 0 else 0
                similarity_diffs.append(relative_diff)
            
            print(f"Combination {current_group}: similarity={current_sim:.6f}, change_rate={similarity_diffs[-1]:.6f}")
        
        return merged_groups, group_similarities, similarity_diffs
    
    def find_optimal_merger(self, merged_groups: List[List[int]], similarities: List[float], diffs: List[float]) -> Tuple[List[int], float]:
        """Find best merge point based on similarity change"""
        if not merged_groups:
            return [], 0.0
            
        # Find significant drop point (negative change rate with large magnitude)
        # Method 1: Find first obvious drop (change rate < -0.1)
        for i, diff in enumerate(diffs[1:], 1):  # Start from second element
            if diff < -0.1:  # Significant drop threshold
                print(f"Detected significant similarity drop at position {i}, change_rate={diff:.6f}")
                return merged_groups[i-1], similarities[i-1]
        
        # Method 2: Find point with minimum change rate (max drop)
        min_diff_idx = np.argmin(diffs)
        if min_diff_idx > 0:  # Ensure not first point
            print(f"Max similarity drop at position {min_diff_idx}, change_rate={diffs[min_diff_idx]:.6f}")
            return merged_groups[min_diff_idx-1], similarities[min_diff_idx-1]
        
        # If no obvious drop, use entire combination
        return merged_groups[-1], similarities[-1]
    
    def process_object(self, object_path: str, text_embeds: torch.Tensor, 
                      mode: str = 'segmentation', save_path: Optional[str] = None, 
                      use_advanced_clustering: bool = False, n_clusters: int = 8) -> Tuple[List[int], float]:
        # Get shape embeddings and point cloud
        from release_tmp.bottle_check.a4_partbasedseman import eval_obj_wild
        shape_embeds, _, shape_pts, _ = eval_obj_wild(
            self.model, object_path, mode, save_path
        )
        
        # Verify obtained features
        print(f"shape_embeds from model: shape={shape_embeds.shape}, device={shape_embeds.device}")
        
        # Move data to device
        shape_embeds = shape_embeds.to(self.device)
        shape_pts = shape_pts.to(self.device)
        text_embeds = text_embeds.to(self.device)
        
        # Cluster to get parts (optionally use new wrapper)
        if use_advanced_clustering:
            seg_labels = self.cluster_parts_with_wrapper(shape_embeds, text_embeds, n_clusters)
        else:
            seg_labels = self.cluster_parts(shape_embeds)
        print(f"Clustering result: unique labels={torch.unique(seg_labels).cpu().numpy()}")
        
        # Get part masks
        part_masks = self.get_part_masks(seg_labels)
        
        # Sort parts by similarity
        sorted_labels, sorted_sims = self.sort_parts_by_similarity(
            part_masks, shape_embeds, text_embeds
        )
        print(f"Parts sorted by similarity: {sorted_labels}")
        print(f"Corresponding similarities: {[f'{s:.6f}' for s in sorted_sims]}")
        
        # Analyze similarity changes during merge
        merged_groups, group_sims, sim_diffs = self.analyze_similarity_changes(
            sorted_labels, part_masks, shape_embeds, text_embeds
        )
        
        # Find best merge point
        best_group, best_sim = self.find_optimal_merger(merged_groups, group_sims, sim_diffs)
        print(f"Best merge combination: {best_group}, similarity={best_sim:.6f}")
        
        if save_path:
            self.visualize_result(shape_pts, seg_labels, best_group, save_path)
        
        return best_group, best_sim
    
    def visualize_result(self, shape_pts: torch.Tensor, seg_labels: torch.Tensor,
                        best_group: List[int], save_path: str):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(shape_pts.cpu().numpy())
        
        target_color = self.hsv_to_rgb(0.6, 0.8, 0.9)  # Blue-ish
        background_color = self.hsv_to_rgb(0, 0, 0.7)  # Gray-ish
        
        colors = []
        best_mask = torch.zeros_like(seg_labels, dtype=bool)
        for label in best_group:
            best_mask |= (seg_labels == label)
        
        for is_target in best_mask.cpu().numpy():
            colors.append(target_color if is_target else background_color)
        
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        save_dir = os.path.dirname(save_path)
        os.makedirs(save_dir, exist_ok=True)
        
        o3d.io.write_point_cloud(save_path, pcd)
        print(f"Visualization result saved to: {save_path}")
