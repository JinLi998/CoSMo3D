"""Single-sample data loading with offline text-embedding cache."""

import os

import torch
from transformers import AutoModel, AutoTokenizer

from app.segment._data import prep_points_val3d
from app.segment.eval_benchmark import get_part_labels


def _load_text_model(textembeds, hf_model_path, device):
    if textembeds == "mpnet":
        model_name = "sentence-transformers/all-mpnet-base-v2"
    else:
        model_name = hf_model_path or "google/siglip-base-patch16-224"

    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    text_model = AutoModel.from_pretrained(model_name, local_files_only=True).eval().to(device)
    return tokenizer, text_model


def _encode_text(labels, textembeds, tokenizer, text_model, device):
    with torch.no_grad():
        inputs = tokenizer(
            labels,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).to(device)

        if textembeds == "mpnet":
            outputs = text_model(**inputs)
            token_embeddings = outputs[0]
            input_mask = inputs["attention_mask"].unsqueeze(-1).expand(token_embeddings.size())
            text_feat = torch.sum(token_embeddings * input_mask, 1) / torch.clamp(input_mask.sum(1), min=1e-9)
        else:
            text_feat = text_model.get_text_features(**inputs)

    text_feat = text_feat / (text_feat.norm(dim=-1, keepdim=True) + 1e-12)
    return text_feat.to(device)


def load_single_sample(
    data_path,
    category,
    textembeds="clip",
    decorated=True,
    hf_model_path=None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return_dict = {}

    mask_pts = torch.load(f"{data_path}/mask2points.pt", map_location=device)
    pts_xyz = torch.load(f"{data_path}/points.pt", map_location=device)
    normal = torch.load(f"{data_path}/normals.pt", map_location=device)
    pts_rgb = torch.load(f"{data_path}/rgb.pt", map_location=device) * 255
    gt = get_part_labels(pts_xyz, mask_pts) + 1
    gt = gt.to(device)

    with open(f"{data_path}/mask_labels.txt", "r", encoding="utf-8") as f:
        labels = [line.strip() for line in f.readlines() if line.strip()]
    if decorated:
        labels = [f"{part} of a {category}" for part in labels]

    text_feat_path = os.path.join(data_path, "text_feat.pt")
    if os.path.exists(text_feat_path):
        text_feat = torch.load(text_feat_path, map_location=device)
    else:
        tokenizer, text_model = _load_text_model(textembeds, hf_model_path, device)
        text_feat = _encode_text(labels, textembeds, tokenizer, text_model, device)
        torch.save(text_feat.cpu(), text_feat_path)

    return_dict = prep_points_val3d(
        pts_xyz.cpu().numpy(),
        pts_rgb.cpu().numpy(),
        normal.cpu().numpy(),
        gt.cpu().numpy(),
        pts_xyz.cpu().numpy(),
        gt.cpu().numpy(),
    )

    for key in return_dict:
        if isinstance(return_dict[key], torch.Tensor):
            return_dict[key] = return_dict[key].to(device)

    return_dict["label_embeds"] = text_feat.to(device)
    return_dict["class_name"] = category
    return_dict["xyz_visualization"] = pts_xyz.float().to(device)
    return_dict["offset"] = return_dict["offset"].to(device)
    return_dict["labels"] = labels
    return return_dict
