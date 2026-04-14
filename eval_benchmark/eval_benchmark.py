"""
Release benchmark evaluation script for CoSMo3D.

Supported benchmarks and default dataset roots (under `dataset/` after HF download):
- 3dcompat200  -> dataset/test_3dcompat200
- partnet      -> dataset/partnet
- shapenetpart -> dataset/test_shapenetpart
"""

import argparse
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from model.data.data import Eval3dcom, EvalPartNetE, EvalShapeNetPart, collate_fn
from model.evaluation.core import compute_3d_iou_upsample
from model.evaluation.utils import load_model, set_seed


PARTNET_CATEGORIES: Dict[str, int] = {
    "Bottle": 16,
    "Box": 17,
    "Bucket": 18,
    "Camera": 19,
    "Cart": 20,
    "Chair": 21,
    "Clock": 22,
    "CoffeeMachine": 23,
    "Dishwasher": 24,
    "Dispenser": 25,
    "Display": 26,
    "Eyeglasses": 27,
    "Faucet": 28,
    "FoldingChair": 29,
    "Globe": 30,
    "Kettle": 31,
    "Keyboard": 32,
    "KitchenPot": 33,
    "Knife": 34,
    "Lamp": 35,
    "Laptop": 36,
    "Lighter": 37,
    "Microwave": 38,
    "Mouse": 39,
    "Oven": 40,
    "Pen": 41,
    "Phone": 42,
    "Pliers": 43,
    "Printer": 44,
    "Refrigerator": 45,
    "Remote": 46,
    "Safe": 47,
    "Scissors": 48,
    "Stapler": 49,
    "StorageFurniture": 50,
    "Suitcase": 51,
    "Switch": 52,
    "Table": 53,
    "Toaster": 54,
    "Toilet": 55,
    "TrashCan": 56,
    "USB": 57,
    "WashingMachine": 58,
    "Window": 59,
    "Door": 60,
}

SHAPENETPART_CATEGORIES: Dict[str, int] = {
    "airplane": 0,
    "bag": 1,
    "cap": 2,
    "car": 3,
    "chair": 4,
    "earphone": 5,
    "guitar": 6,
    "knife": 7,
    "lamp": 8,
    "laptop": 9,
    "motorbike": 10,
    "mug": 11,
    "pistol": 12,
    "rocket": 13,
    "skateboard": 14,
    "table": 15,
}

DEFAULT_DATA_ROOTS = {
    "3dcompat200": "dataset/test_3dcompat200",
    "partnet": "dataset/partnet",
    "shapenetpart": "dataset/test_shapenetpart",
}

BENCHMARK_ALIASES = {
    "3dcompat200": "3dcompat200",
    "d3compat": "3dcompat200",
    "partnet": "partnet",
    "partnete": "partnet",
    "shapenetpart": "shapenetpart",
    "shapnetpart": "shapenetpart",
}


def evaluate_loader(
    model: torch.nn.Module,
    dataloader: DataLoader,
    *,
    panoptic: bool,
    n_chunks: int,
) -> float:
    """Evaluate one dataloader and return mean IoU."""
    temperature = np.exp(model.ln_logit_scale.item())
    ious: List[float] = []

    with torch.no_grad():
        for data in dataloader:
            for key, value in data.items():
                if isinstance(value, torch.Tensor) and "full" not in key:
                    data[key] = value.cuda(non_blocking=True)

            net_out = model(x=data)
            full_miou, _ = compute_3d_iou_upsample(
                net_out,
                data["label_embeds"],
                temperature,
                data["class_name"][0],
                data["coord"],
                data["xyz_full"],
                data["gt_full"],
                panoptic=panoptic,
                N_CHUNKS=n_chunks,
                visualize_seg=False,
                savepath=None,
                visualize_all_heatmap=False,
                xyz_visualization=data["xyz_visualization"],
            )
            print(f"{data['class_name'][0]}: {full_miou:.4f}")
            ious.append(full_miou)

    return float(np.mean(ious)) if ious else 0.0


def _build_loader(dataset) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        drop_last=False,
    )


def evaluate_3dcompat200(
    model: torch.nn.Module,
    data_root: str,
    split: Optional[str],
    apply_rotation: bool,
    decorated: bool,
    use_tuned_prompt: bool,
    d3com_datatype: str,
) -> None:
    del split, use_tuned_prompt  # kept for CLI compatibility
    categories = sorted([x for x in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, x))])
    if not categories:
        raise ValueError(f"No category folders found under '{data_root}' for 3dcompat200 evaluation.")

    rows: List[Tuple[str, float]] = []
    durations: List[float] = []
    for category in categories:
        dataset = Eval3dcom(
            data_root=data_root,
            category=category,
            datatype=d3com_datatype,
            apply_rotation=apply_rotation,
            decorated=decorated,
        )
        if len(dataset) == 0:
            continue

        loader = _build_loader(dataset)
        start = time.time()
        # Align with pipes_eval/d3compat evaluation.
        miou = evaluate_loader(model, loader, panoptic=False, n_chunks=20)
        elapsed = time.time() - start
        rows.append((category, miou))
        durations.append(elapsed)
        print(f"[3dcompat200/{d3com_datatype}/{category}] mIoU: {miou:.4f} | time: {elapsed:.2f}s")

    if not rows:
        raise ValueError(
            f"No instances matched datatype='{d3com_datatype}'. "
            "Expected instance names containing 'coarse' or 'fine' under each category."
        )

    avg_miou = float(np.mean([x[1] for x in rows]))
    total_time = float(np.sum(durations))
    print(f"\n[3dcompat200/{d3com_datatype}] average mIoU: {avg_miou:.4f} | total time: {total_time:.2f}s")


def evaluate_partnet(
    model: torch.nn.Module,
    data_root: str,
    apply_rotation: bool,
    subset: bool,
    decorated: bool,
    save_dir: Optional[str],
) -> None:
    rows: List[Tuple[str, float]] = []
    durations: List[float] = []

    for category in PARTNET_CATEGORIES:
        dataset = EvalPartNetE(
            data_root=data_root,
            category=category,
            apply_rotation=apply_rotation,
            subset=subset,
            decorated=decorated,
        )
        loader = _build_loader(dataset)
        start = time.time()
        miou = evaluate_loader(model, loader, panoptic=False, n_chunks=20)
        elapsed = time.time() - start
        rows.append((category, miou))
        durations.append(elapsed)
        print(f"[partnet/{category}] mIoU: {miou:.4f} | time: {elapsed:.2f}s")

    avg_miou = float(np.mean([x[1] for x in rows])) if rows else 0.0
    total_time = float(np.sum(durations)) if durations else 0.0

    print("\nCategory          mIoU")
    print("-" * 28)
    for category, miou in rows:
        print(f"{category:<16}  {miou:.4f}")
    print("-" * 28)
    print(f"{'Average':<16}  {avg_miou:.4f}")
    print(f"Total time: {total_time:.2f}s")

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        out_path = os.path.join(save_dir, "result_table.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("Category          mIoU\n")
            f.write("-" * 28 + "\n")
            for category, miou in rows:
                f.write(f"{category:<16}  {miou:.4f}\n")
            f.write("-" * 28 + "\n")
            f.write(f"{'Average':<16}  {avg_miou:.4f}\n")
            f.write(f"Total time: {total_time:.2f}s\n")
        print(f"Saved results to: {out_path}")


def evaluate_shapenetpart(
    model: torch.nn.Module,
    data_root: str,
    apply_rotation: bool,
    subset: bool,
    decorated: bool,
    use_tuned_prompt: bool,
) -> None:
    rows: List[Tuple[str, float]] = []
    durations: List[float] = []

    for category in SHAPENETPART_CATEGORIES:
        dataset = EvalShapeNetPart(
            data_path=data_root,
            class_choice=category,
            apply_rotation=apply_rotation,
            subset=subset,
            decorated=decorated,
            use_tuned_prompt=use_tuned_prompt,
        )
        loader = _build_loader(dataset)
        start = time.time()
        miou = evaluate_loader(model, loader, panoptic=True, n_chunks=1)
        elapsed = time.time() - start
        rows.append((category, miou))
        durations.append(elapsed)
        print(f"[shapenetpart/{category}] mIoU: {miou:.4f} | time: {elapsed:.2f}s")

    avg_miou = float(np.mean([x[1] for x in rows])) if rows else 0.0
    total_time = float(np.sum(durations)) if durations else 0.0
    print(f"\n[shapenetpart] average mIoU: {avg_miou:.4f} | total time: {total_time:.2f}s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CoSMo3D release benchmark evaluation")
    parser.add_argument(
        "--benchmark",
        required=True,
        type=str,
        help="Benchmark to evaluate (3dcompat200/d3compat, partnet/partnete, shapenetpart/shapnetpart).",
    )
    parser.add_argument(
        "--checkpoint_path",
        required=True,
        help="Path to model checkpoint.",
    )
    parser.add_argument(
        "--data_root",
        default=None,
        help="Dataset root path. If not set, uses benchmark default path.",
    )
    parser.add_argument(
        "--save_dir",
        default="results",
        help="Directory to save summary tables (used by partnet).",
    )
    parser.add_argument(
        "--split",
        default=None,
        help="Optional split name for legacy Objaverse-style layout (seenclass/unseen/shapenetpart).",
    )
    parser.add_argument(
        "--d3com_datatype",
        default="coarse",
        choices=["coarse", "fine"],
        help="3DCompat subset, aligned with pipes_eval/eval_d3compat.sh.",
    )
    parser.add_argument(
        "--rotate",
        action="store_true",
        help="Apply predefined random rotations for partnet/shapenetpart.",
    )
    parser.add_argument(
        "--canonical",
        action="store_false",
        dest="rotate",
        help="Compatibility flag from pipes_eval. Keeps canonical orientation (no random rotation).",
    )
    parser.add_argument(
        "--subset",
        action="store_true",
        help="Evaluate only predefined subset when supported by dataset loader.",
    )
    parser.add_argument(
        "--plain_prompt",
        action="store_true",
        help="Use plain part prompts instead of 'part of a object'.",
    )
    parser.add_argument(
        "--part_query",
        action="store_true",
        help="Compatibility flag from pipes_eval. Use plain part names as prompts.",
    )
    parser.add_argument(
        "--use_shapenetpart_topk_prompt",
        action="store_true",
        help="Use tuned prompts for shapenetpart settings.",
    )
    return parser.parse_args()


def main() -> None:
    set_seed(123)
    args = parse_args()
    benchmark_key = BENCHMARK_ALIASES.get(args.benchmark.lower())
    if benchmark_key is None:
        valid = ", ".join(sorted(BENCHMARK_ALIASES.keys()))
        raise ValueError(f"Unsupported benchmark '{args.benchmark}'. Supported values: {valid}")

    if args.data_root:
        data_root = args.data_root
    else:
        data_root = DEFAULT_DATA_ROOTS[benchmark_key]

    decorated = not (args.plain_prompt or args.part_query)

    print(f"Benchmark: {benchmark_key}")
    print(f"Data root: {data_root}")
    print(f"Checkpoint: {args.checkpoint_path}")

    model = load_model(args.checkpoint_path)

    if benchmark_key == "3dcompat200":
        evaluate_3dcompat200(
            model=model,
            data_root=data_root,
            split=args.split,
            apply_rotation=args.rotate,
            decorated=decorated,
            use_tuned_prompt=args.use_shapenetpart_topk_prompt,
            d3com_datatype=args.d3com_datatype,
        )
    elif benchmark_key == "partnet":
        evaluate_partnet(
            model=model,
            data_root=data_root,
            apply_rotation=args.rotate,
            subset=args.subset,
            decorated=decorated,
            save_dir=args.save_dir,
        )
    else:
        evaluate_shapenetpart(
            model=model,
            data_root=data_root,
            apply_rotation=args.rotate,
            subset=args.subset,
            decorated=decorated,
            use_tuned_prompt=args.use_shapenetpart_topk_prompt,
        )


if __name__ == "__main__":
    main()
