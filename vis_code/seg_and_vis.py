"""
Segment a single 3DCoMPaT sample and visualize results on a GLB mesh.

Usage (from project root, conda env find3d):
    python -m vis_code.seg_and_vis

Reference:
    - Segmentation: app/segment/eval_benchmark.py
    - Mesh visualization: cosmo3d/eval2ivs_v2/segvis/c4_objlabs.py
"""

import argparse
import os
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import torch

from model.evaluation.utils import load_model
from release_module.network.canoncolor_bbox_pre import PointSemSegWithDecoder
from vis_code.mesh_vis import DEFAULT_COLOR_MAP, export_colored_glb, load_mesh, transfer_labels_to_mesh
from vis_code.sample_data import load_single_sample


def resolve_path(project_root, path):
    if os.path.isabs(path):
        return path
    return os.path.join(project_root, path)


def run_segmentation(model, data, n_chunks=20):
    """Run model inference and upsample labels to the full point cloud."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    temperature = np.exp(model.ln_logit_scale.item()) if hasattr(model, "ln_logit_scale") else 1.0

    with torch.no_grad():
        data["mask_offset"] = torch.tensor([data["label_embeds"].shape[0]], device=device)
        model_output = model(data)

        if isinstance(model_output, (tuple, list)):
            net_out = model_output[0]
        elif isinstance(model_output, torch.Tensor):
            net_out = model_output
        else:
            raise TypeError(f"Unsupported model output format: {type(model_output)}")

    logits = net_out @ data["label_embeds"].t() * temperature
    pred_labels = torch.argmax(logits, dim=1) + 1

    xyz_sub = data["coord"]
    xyz_full = data["xyz_full"].squeeze()
    chunk_len = xyz_full.shape[0] // n_chunks + 1
    closest_idx_list = []

    for i in range(n_chunks):
        cur_chunk = xyz_full[chunk_len * i : chunk_len * (i + 1)].to(device)
        dist = torch.norm(xyz_sub.unsqueeze(0) - cur_chunk.unsqueeze(1), dim=-1)
        min_idxs = torch.min(dist, dim=1)[1]
        closest_idx_list.append(min_idxs)

    all_nn_idxs = torch.cat(closest_idx_list, axis=0)
    pred_full = pred_labels[all_nn_idxs].cpu().numpy()
    return pred_full, temperature


def save_point_seg_results(output_dir, sample_name, pred_full, part_names, color_map):
    os.makedirs(output_dir, exist_ok=True)
    seg_txt_path = os.path.join(output_dir, f"{sample_name}_seg.txt")
    np.savetxt(seg_txt_path, pred_full, fmt="%d")

    color_txt_path = os.path.join(output_dir, f"{sample_name}_color_semantic.txt")
    with open(color_txt_path, "w", encoding="utf-8") as f:
        for idx, name in enumerate(part_names, start=1):
            rgb = color_map.get(idx, DEFAULT_COLOR_MAP[0])
            f.write(f"{rgb}, {name} (label: {idx})\n")

    return seg_txt_path, color_txt_path


def default_project_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser(description="Segment one sample and export colored GLB.")
    parser.add_argument(
        "--project_root",
        type=str,
        default=None,
        help="Project root directory. Defaults to the parent of vis_code/.",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="dataset/checkpoints/ours_final.pth",
        help="Checkpoint path relative to project_root (or absolute).",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="data_test/coarse_b'29_0cb'",
        help="Sample directory relative to project_root.",
    )
    parser.add_argument(
        "--mesh_path",
        type=str,
        default="data_test/29_0cb.glb",
        help="Target GLB mesh relative to project_root.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Output directory relative to project_root.",
    )
    parser.add_argument("--category", type=str, default="vase", help="Object category for text prompts.")
    parser.add_argument("--net_type", type=str, default="net8", help="Network type.")
    parser.add_argument("--textembeds", type=str, default="clip", choices=["clip", "mpnet"])
    parser.add_argument(
        "--hf_model_path",
        type=str,
        default=None,
        help="Local HuggingFace model id or snapshot path. Defaults to google/siglip-base-patch16-224.",
    )
    parser.add_argument("--decorated", action="store_true", default=True)
    parser.add_argument("--plain_prompt", action="store_true", help="Use plain part names without category decoration.")
    parser.add_argument("--n_chunks", type=int, default=20, help="Chunks for nearest-neighbor upsampling.")
    parser.add_argument("--sample_name", type=str, default="29_0cb", help="Output file prefix.")
    parser.add_argument(
        "--no_mesh_align",
        action="store_true",
        help="Disable +90deg X-axis rotation before point-to-face label transfer.",
    )
    args = parser.parse_args()

    project_root = os.path.abspath(args.project_root or default_project_root())
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    os.chdir(project_root)

    data_path = resolve_path(project_root, args.data_path)
    mesh_path = resolve_path(project_root, args.mesh_path)
    output_dir = resolve_path(project_root, args.output_dir)
    checkpoint_path = resolve_path(project_root, args.checkpoint_path)

    for path, name in [
        (data_path, "data_path"),
        (mesh_path, "mesh_path"),
        (checkpoint_path, "checkpoint_path"),
    ]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{name} does not exist: {path}")

    torch.manual_seed(123)
    if args.net_type in ["net1", "net2"]:
        model = load_model(checkpoint_path)
    else:
        model = PointSemSegWithDecoder(args=args)
        model.load_state_dict(torch.load(checkpoint_path, map_location="cpu")["model_state_dict"], strict=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.eval().to(device)

    decorated = args.decorated and not args.plain_prompt
    data = load_single_sample(
        data_path,
        args.category,
        args.textembeds,
        decorated=decorated,
        hf_model_path=args.hf_model_path,
    )

    for key, value in data.items():
        if isinstance(value, torch.Tensor):
            data[key] = value.to(device)

    pred_full, temperature = run_segmentation(model, data, n_chunks=args.n_chunks)

    with open(os.path.join(data_path, "mask_labels.txt"), "r", encoding="utf-8") as f:
        part_names = [line.strip() for line in f.readlines() if line.strip()]

    points = torch.load(os.path.join(data_path, "points.pt"), map_location="cpu").numpy().astype(np.float64)
    if len(pred_full) != len(points):
        raise ValueError(f"Label count {len(pred_full)} does not match point count {len(points)}")

    seg_txt_path, color_txt_path = save_point_seg_results(
        output_dir, args.sample_name, pred_full, part_names, DEFAULT_COLOR_MAP
    )

    mesh = load_mesh(mesh_path)
    face_labels = transfer_labels_to_mesh(
        mesh, points, pred_full, align_to_points=not args.no_mesh_align
    )
    output_glb = os.path.join(output_dir, f"{args.sample_name}_seg.glb")
    export_colored_glb(mesh, face_labels, output_glb, color_map=DEFAULT_COLOR_MAP)

    face_label_path = os.path.join(output_dir, f"{args.sample_name}_face_labels.txt")
    np.savetxt(face_label_path, face_labels, fmt="%d")

    print("Segmentation and visualization finished.")
    print(f"  data_path      : {data_path}")
    print(f"  mesh_path      : {mesh_path}")
    print(f"  checkpoint     : {checkpoint_path}")
    print(f"  category       : {args.category}")
    print(f"  temperature    : {temperature:.4f}")
    print(f"  point labels   : {seg_txt_path}")
    print(f"  color mapping  : {color_txt_path}")
    print(f"  face labels    : {face_label_path}")
    print(f"  colored glb    : {output_glb}")


if __name__ == "__main__":
    main()
