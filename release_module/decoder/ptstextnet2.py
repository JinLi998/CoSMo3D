### Batch * data size; feats size organized via offset and mask_offset recording each batch end index. Accelerated version without iteration.
### Further modified so one text embedding corresponds to one output.

from dataclasses import dataclass
import math
import torch
import torch.nn as nn
from einops import rearrange  # Remove repeat dependency, keep only necessary ops

from release_module.decoder.transformers.perceiver_1d import Perceiver
from release_module.decoder.transformers.attention import ResidualCrossAttentionBlock
from release_module.decoder.utils.checkpoint import checkpoint
from release_module.decoder.utils.base import BaseModule

from release_module.decoder.autoencoders.michelangelo_autoencoder import get_embedder








################### Canonical color and bbox prediction (decoder segmentation removed)
@dataclass
class PointCloudTextCanoncolorWodecoderTransformerConfig(BaseModule.Config):
    # Point cloud config
    point_cloud_embed_type: str = "fourier"
    point_cloud_num_freqs: int = 8  # 6
    point_cloud_include_pi: bool = False  # True
    
    # Model dimension config
    feature_dim: int = 768  # Feature dim for point cloud and text
    width: int = 768        # Model hidden layer dimension
    heads: int = 12         # Number of attention heads
    num_self_attn_layers: int = 16  # Number of Self Attention layers
    
    # Attention config
    init_scale: float = 0.25
    qkv_bias: bool = False  # True
    use_flash: bool = True  # False
    use_checkpoint: bool = False
    
    # Output config
    output_dim: int = 1  # Prediction output dim (probability)
    color_output_dim: int = 3  # Color prediction output dim (R, G, B)
    bbox_output_dim: int = 6  # 3D BBox prediction output dim (x_min, y_min, z_min, x_max, y_max, z_max)


class PointCloudTextCanoncolorWodecoderTransformer(BaseModule):
    Config = PointCloudTextCanoncolorWodecoderTransformerConfig
    
    def __init__(self, cfg=None):
        super().__init__(cfg)
    
    def configure(self) -> None:
        super().configure()
        
        # Initialize point cloud coordinate embedder
        self.pc_embedder = get_embedder(
            embed_type=self.cfg.point_cloud_embed_type,
            num_freqs=self.cfg.point_cloud_num_freqs,
            input_dim=3,
            include_pi=self.cfg.point_cloud_include_pi
        )
        
        # Point cloud feature projection layer
        if hasattr(self.pc_embedder, 'out_dim'):
            embed_dim = self.pc_embedder.out_dim
        else:
            # If embedder has no out_dim, compute from freq count (preserve original logic)
            embed_dim = 3 * (2 ** self.cfg.point_cloud_num_freqs * 2)
        
        self.pc_feature_proj = nn.Linear(
            embed_dim + self.cfg.feature_dim, 
            self.cfg.width
        )
        
        # Text feature projection layer
        self.text_feature_proj = nn.Linear(
            self.cfg.feature_dim, 
            self.cfg.width
        )

        # First Cross Attention
        self.first_cross_attn = ResidualCrossAttentionBlock(
            width=self.cfg.width,
            heads=self.cfg.heads,
            init_scale=self.cfg.init_scale * math.sqrt(1.0 / self.cfg.width),
            qkv_bias=self.cfg.qkv_bias,
            use_flash=self.cfg.use_flash,
        )
        
        # # Self Attention layers
        # self.self_attn = Perceiver(
        #     n_ctx=None,
        #     width=self.cfg.width,
        #     layers=self.cfg.num_self_attn_layers,
        #     heads=self.cfg.heads,
        #     init_scale=self.cfg.init_scale * math.sqrt(1.0 / self.cfg.width),
        #     qkv_bias=self.cfg.qkv_bias,
        #     use_flash=self.cfg.use_flash,
        #     use_checkpoint=self.cfg.use_checkpoint
        # )
        
        # # Second Cross Attention
        # self.second_cross_attn = ResidualCrossAttentionBlock(
        #     width=self.cfg.width,
        #     heads=self.cfg.heads,
        #     init_scale=self.cfg.init_scale * math.sqrt(1.0 / self.cfg.width),
        #     qkv_bias=self.cfg.qkv_bias,
        #     use_flash=self.cfg.use_flash,
        # )
        
        # Output projection and LayerNorm (probability prediction)
        # self.output_proj = nn.Linear(self.cfg.width, self.cfg.output_dim)
        # self.ln_post = nn.LayerNorm(self.cfg.width)
        
        # Color prediction head (predict RGB 3 channels)
        self.color_head = nn.Sequential(
            nn.LayerNorm(self.cfg.width),
            nn.Linear(self.cfg.width, self.cfg.width),
            nn.ReLU(),
            nn.Linear(self.cfg.width, 3)  # Directly predict RGB 3 channels
        )
        
        # 3D BBox prediction branch - extract global point cloud features
        self.global_pool = nn.AdaptiveAvgPool1d(1)  # Global avg pool for point cloud features
        
        # 3D BBox prediction branch - feature fusion and prediction layers
        self.bbox_feature_fusion = nn.Linear(3 * self.cfg.width, self.cfg.width)
        self.bbox_proj1 = nn.Linear(self.cfg.width, self.cfg.width)
        self.bbox_act = nn.ReLU()
        self.bbox_proj2 = nn.Linear(self.cfg.width, self.cfg.bbox_output_dim)
    

    def _forward(self, 
                point_cloud: torch.FloatTensor,  # shape: [total_points, 3]
                point_features: torch.FloatTensor,  # shape: [total_points, feat_dim]
                text_features: torch.FloatTensor,  # shape: [total_texts, text_feat_dim]
                offset: torch.LongTensor,  # shape: [batch_size] (point cloud batch end indices)
                mask_offset: torch.LongTensor  # shape: [batch_size] (text group end indices)
                ) :  
        # Returns: (prob output, color output, prob offset, color offset, bbox pred, bbox offset)
        batch_size = offset.size(0)
        device = point_cloud.device
        dtype = offset.dtype
        
        # --------------------------
        # 1. Preprocess: align point cloud batches with text groups
        # --------------------------
        # Point cloud offsets (incl. start 0): [0, p0_end, p1_end, ..., pB_end] → len=B+1
        point_offsets = torch.cat([torch.tensor([0], device=device, dtype=dtype), offset])
        # Text group offsets (incl. start 0): [0, t0_end, t1_end, ..., tB_end] → len=B+1
        text_group_offsets = torch.cat([torch.tensor([0], device=device, dtype=dtype), mask_offset])
        
        # Compute each point cloud batch length: [p0_len, p1_len, ..., pB_len] → len=B
        point_batch_lens = point_offsets[1:] - point_offsets[:-1]
        # Compute text count per group: [t0_cnt, t1_cnt, ..., tB_cnt] → len=B
        text_group_cnts = text_group_offsets[1:] - text_group_offsets[:-1]
        
        # Handle empty input
        if batch_size == 0:
            return (torch.empty(0, self.cfg.output_dim, device=device),
                    torch.empty(0, 3, device=device),  # Color output fixed to 3 channels (RGB)
                    torch.empty(0, dtype=dtype, device=device),
                    torch.empty(0, dtype=dtype, device=device),
                    torch.empty(0, self.cfg.bbox_output_dim, device=device),
                    torch.empty(0, dtype=dtype, device=device))
        
        # --------------------------
        # 2. Build text-to-point-batch mapping (text -> point cloud batch)“text-text”        # --------------------------
        # Example: batch 0 has 3 texts, batch 1 has 2 texts → text_to_point_batch = [0,0,0,1,1]
        text_to_point_batch = []
        for i in range(batch_size):
            text_to_point_batch.extend([i] * text_group_cnts[i].item())
        text_to_point_batch = torch.tensor(text_to_point_batch, device=device, dtype=dtype)  # [total_texts]
        total_texts = text_to_point_batch.size(0)
        
        # --------------------------
        # 3. Vectorization prep: expand point cloud to match text count
        # --------------------------
        # 3.1 Extract raw data per point cloud batch
        point_batch_list = []  # Store [p_len, 3] per point cloud batch
        point_feat_batch_list = []  # Store [p_len, feat_dim] per point cloud batch
        for i in range(batch_size):
            p_start, p_end = point_offsets[i], point_offsets[i+1]
            point_batch_list.append(point_cloud[p_start:p_end])
            point_feat_batch_list.append(point_features[p_start:p_end])
        
        # 3.2 Expand point cloud by text mapping: each text gets full point cloud of its batch
        # Output shape: [total_texts, max_p_len, 3] (max_p_len = max length over batches)
        max_p_len = max([p.size(0) for p in point_batch_list]) if point_batch_list else 0
        expanded_point = torch.zeros(total_texts, max_p_len, 3, device=device, dtype=point_cloud.dtype)
        expanded_point_feat = torch.zeros(total_texts, max_p_len, point_features.size(1), 
                                        device=device, dtype=point_features.dtype)
        expanded_point_mask = torch.zeros(total_texts, max_p_len, device=device, dtype=torch.bool)  # Valid point mask
        
        for text_idx in range(total_texts):
            point_batch_idx = text_to_point_batch[text_idx].item()  # Point cloud batch for this text
            p_data = point_batch_list[point_batch_idx]  # [p_len, 3]
            p_feat = point_feat_batch_list[point_batch_idx]  # [p_len, feat_dim]
            p_len = p_data.size(0)
            
            # Fill point cloud data for this text
            expanded_point[text_idx, :p_len] = p_data
            expanded_point_feat[text_idx, :p_len] = p_feat
            expanded_point_mask[text_idx, :p_len] = True  # Mark valid points
        
        # --------------------------
        # 4. Text feature processing: use raw text features (no padding)
        # --------------------------
        # text_features is [total_texts, text_feat_dim], one row per text
        text_proj = self.text_feature_proj(text_features)  # [total_texts, width]
        # Add dim to match attention input (attention expects [B, seq_len, dim])
        text_proj = text_proj.unsqueeze(1)  # [total_texts, 1, width] (each text as len-1 sequence)
        
        # --------------------------
        # 5. Point cloud feature processing
        # --------------------------
        # Point cloud embedding: [total_texts, max_p_len, embed_dim]
        point_embedded = self.pc_embedder(expanded_point)
        # Concatenate point cloud features: [total_texts, max_p_len, embed_dim + feat_dim]
        point_feat = torch.cat([point_embedded, expanded_point_feat], dim=-1)
        # Point cloud projection: [total_texts, max_p_len, width]
        point_proj = self.pc_feature_proj(point_feat)
        # Mask invalid points: [total_texts, max_p_len, width]
        point_proj = point_proj * expanded_point_mask.unsqueeze(-1)
        
        # --------------------------
        # 6. Attention (core: each text interacts with its point cloud) - for prob prediction only
        # --------------------------
        # 6.1 First cross attention: text (query) → point cloud (context)
        # Input: query=[total_texts, 1, width], context=[total_texts, max_p_len, width]
        cross1_out = self.first_cross_attn(text_proj, point_proj)  # [total_texts, 1, width]
        # Mask invalid results (texts are all valid here, optional but consistent)
        cross1_out = cross1_out * torch.ones_like(expanded_point_mask[:, :1], dtype=torch.float32).unsqueeze(-1)
        
        # 6.2 Text self attention: [total_texts, 1, width] → seq len 1, self-attn trivial, can skip
        # self_attn_out = self.self_attn(cross1_out)  # [total_texts, 1, width]
        
        # # 6.3 Second cross attention: point cloud (query) → text (context)
        # # Input: query=[total_texts, max_p_len, width], context=[total_texts, 1, width]
        # cross2_out = self.second_cross_attn(point_proj, self_attn_out)  # [total_texts, max_p_len, width]
        # # Mask invalid points
        # cross2_out = cross2_out * expanded_point_mask.unsqueeze(-1)
        
        # --------------------------
        # 7. Color prediction branch (point cloud only, no text)
        # --------------------------
        # 7.1 Aggregate point cloud features per object
        object_point_feats = []
        for obj_idx in range(batch_size):
            # Find all text indices belonging to current object
            obj_text_mask = (text_to_point_batch == obj_idx)
            if obj_text_mask.sum() == 0:
                # Handle objects with no text
                obj_feat = torch.zeros(1, max_p_len, self.cfg.width, device=device)
            else:
                # Aggregate point cloud features for this object (point cloud only)
                obj_feat = point_proj[obj_text_mask].mean(dim=0, keepdim=True)  # [1, max_p_len, width]
            object_point_feats.append(obj_feat)
        object_point_feats = torch.cat(object_point_feats, dim=0)  # [batch_size, max_p_len, width]
        
        # 7.2 Point-level color prediction - predict RGB 3 channels
        color_output_raw = self.color_head(object_point_feats)  # [batch_size, max_p_len, 3]
        color_output_raw = torch.sigmoid(color_output_raw)  # Normalize to 0-1
        
        # 7.3 Extract valid color predictions (1-to-1 with point cloud)
        valid_color_output_list = []
        for obj_idx in range(batch_size):
            p_len = point_batch_lens[obj_idx].item()
            # Extract color for all points of this object
            obj_color_output = color_output_raw[obj_idx, :p_len]  # [p_len, 3]
            valid_color_output_list.append(obj_color_output)
        
        # 7.4 Concatenate all valid color predictions (total length = total points)
        final_color_output = torch.cat(valid_color_output_list, dim=0)  # [total_points, 3]
        
        # 7.5 Generate color output offset (match input point cloud offset)
        color_offset = offset  # Color output 1-to-1 with point cloud, same offset
        # --------------------------
        # 10. Generate new offset for prob output
        # --------------------------
        # Compute length per combination (i.e. corresponding point cloud batch length)
        combo_lens = []
        for i in range(batch_size):
            combo_lens.extend([point_batch_lens[i].item()] * text_group_cnts[i].item())
        # Cumulative sum as new offset: [total_texts]
        prob_offset = torch.cumsum(torch.tensor(combo_lens, device=device, dtype=dtype), dim=0)
        
        # --------------------------
        # 11. 3D BBox prediction branch: fuse point_proj and cross1_out
        # --------------------------
        # 1. Global object features: point_proj_global (global pool, keep context)
        point_proj_global = self.global_pool(point_proj.transpose(1, 2)).squeeze(2)  # [total_texts, width]
        
        # 2. Part features: cross1_features (text-attended part region)
        cross1_features = cross1_out.squeeze(1)  # [total_texts, width]
        
        # 3. Compute global-part relative position encoding
        relative_pos_feat = cross1_features - point_proj_global  # [total_texts, width]
        
        # 4. Strengthen global-part relation: compute attention weight
        part_vs_global_attn = torch.sigmoid(torch.bmm(
            cross1_features.unsqueeze(1),  # [total_texts, 1, width]
            point_proj_global.unsqueeze(2)  # [total_texts, width, 1]
        )).squeeze()  # [total_texts] → part-global affinity (0~1)
        
        # 5. Weighted fusion of base features
        weighted_global = point_proj_global * part_vs_global_attn.unsqueeze(-1)  # [total_texts, width]
        weighted_part = cross1_features * (1 - part_vs_global_attn.unsqueeze(-1))  # [total_texts, width]

        # 6. Concatenate: fused base features + relative position encoding
        fused_features = torch.cat([
            weighted_global, 
            weighted_part, 
            relative_pos_feat  # Relative position features
        ], dim=1)  # [total_texts, 3*width]
        
        # 7. Unify feature dim via fusion layer
        fused_features = self.bbox_feature_fusion(fused_features)  # [total_texts, width]
        
        # 8. Predict part BBox params
        bbox_pred = self.bbox_proj2(self.bbox_act(self.bbox_proj1(fused_features)))
        
        # 8.5 Generate BBox prediction offset
        bbox_offset = torch.arange(1, total_texts + 1, device=device, dtype=dtype) if total_texts > 0 else \
                      torch.empty(0, dtype=dtype, device=device)
        
        # --------------------------
        # 12. Consistency check
        # --------------------------
        total_points = point_cloud.size(0)
        assert final_color_output.size(0) == total_points, \
            f"Color output length {final_color_output.size(0)} != point cloud size {total_points}"
        assert color_offset.size(0) == batch_size, \
            f"Color offset length {color_offset.size(0)} != batch size {batch_size}"
        # assert final_prob_output.size(0) == prob_offset[-1].item() if total_texts > 0 else 0, \
        #     f"Prob output length {final_prob_output.size(0)} != prob offset total {prob_offset[-1].item() if total_texts>0 else 0}"
        assert prob_offset.size(0) == total_texts, \
            f"Prob offset length {prob_offset.size(0)} != total texts {total_texts}"
        assert bbox_pred.size(0) == total_texts, \
            f"BBox pred count {bbox_pred.size(0)} != total texts {total_texts}"
        assert bbox_offset.size(0) == total_texts, \
            f"BBox offset length {bbox_offset.size(0)} != total texts {total_texts}"
        
        return 1, final_color_output, prob_offset, bbox_pred, bbox_offset
        # return 1, 1, prob_offset, bbox_pred, bbox_offset
    
    def forward(self, 
                point_cloud: torch.FloatTensor,
                point_features: torch.FloatTensor,
                text_features: torch.FloatTensor,
                offset: torch.LongTensor,
                mask_offset: torch.LongTensor):
        if self.cfg.use_checkpoint and self.training:
            return self._forward(point_cloud, point_features, text_features, offset, mask_offset)
        else:
            return self._forward(point_cloud, point_features, text_features, offset, mask_offset)
