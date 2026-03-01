"""
Merge the previous network architecture with the new decoder.
"""

import torch
from dataclasses import asdict


from model.backbone.pt3.model import PointSemSeg  # Original backbone
from release_module.decoder.ptstextnet2 import PointCloudTextCanoncolorWodecoderTransformer, PointCloudTextCanoncolorWodecoderTransformerConfig  # Decoder with bbox and canonical color


class PointSemSegWithDecoder(torch.nn.Module):
    """Full model integrating backbone and decoder"""
    def __init__(self, args):
        super().__init__()
        # 1. Initialize pretrained backbone (original PointSemSeg)
        self.backbone = PointSemSeg(args=args, dim_output=768)  # Preserve original output dimension
        # 2. Initialize decoder
        cfg = PointCloudTextCanoncolorWodecoderTransformerConfig()
        cfg_dict = asdict(cfg)
        self.decoder = PointCloudTextCanoncolorWodecoderTransformer(cfg_dict)
        # 3. Keep temperature param from original model (if needed)
        self.ln_logit_scale = self.backbone.ln_logit_scale  # Inherit from backbone

    def forward(self, data):
        # 1. Extract features from backbone
        backbone_feat = self.backbone(data)  # Get 768-dim features
        # return backbone_feat
        
        # 2. Decoder processes features
        decoder_out, canoncolor_out, decoder_offset, bbox_pred, bbox_offset  = self.decoder(data['coord'], backbone_feat,
                                   data['label_embeds'], data['offset'], data['mask_offset'])  # Get decoder output
        
        # print('backbone_feat:', backbone_feat.shape)
        # print('data:', data.keys())
        # print('coord:', data['coord'].shape)
        # print('label_embeds:', data['label_embeds'].shape)
        # print('offset:', len(data['offset']))
        # print('mask_offset:', len(data['mask_offset']))
        # print('decoder_out:', decoder_out.shape)
        # print('decoder_out max:', torch.max(decoder_out))
        # print('decoder_out min:', torch.min(decoder_out))
        # print('canoncolor_out:', canoncolor_out.shape)
        # print('canoncolor max:', torch.max(canoncolor_out))
        # print('canoncolor min:', torch.min(canoncolor_out))
        # print('decoder_offset:', decoder_offset.shape)
        # print('decoder_out:', torch.max(decoder_out), torch.min(decoder_out))
        # asdf
        return backbone_feat, decoder_out, canoncolor_out, decoder_offset, bbox_pred, bbox_offset  # Return intermediate features and final output
