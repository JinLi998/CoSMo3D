"""
Release training script for CoSMo3D canonical-align-bbox setting.

This script is a cleaned version based on pipes/ours_canon_align_bbox:
- keeps core multi-GPU training (DDP)
- keeps baseline + canonical-color + bbox losses
- removes debug-only branches and redundant utilities
"""

import argparse
import os
import random
from typing import Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from model.training.loss import DistillLossContrastive
from release_module.network.canoncolor_bbox_pre import PointSemSegWithDecoder
from release_module.training.data_release import TrainingData, collate_fn
from release_module.training.loss_bbox import PointBasedBBoxOffsetLoss
from release_module.training.loss_canonical_color import CanonicalColorLoss


def setup_ddp(rank: int, world_size: int, port: int) -> None:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_ddp() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def seed_everything(seed: int, rank: int) -> None:
    full_seed = seed + rank
    random.seed(full_seed)
    np.random.seed(full_seed)
    torch.manual_seed(full_seed)
    torch.cuda.manual_seed_all(full_seed)


def create_train_loader(
    data_root: str,
    batch_size: int,
    num_workers: int,
    use_ddp: bool,
) -> Tuple[DataLoader, Optional[DistributedSampler]]:
    dataset = TrainingData(data_root=data_root)
    sampler = DistributedSampler(dataset, shuffle=True) if use_ddp else None
    loader = DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)),
        shuffle=(sampler is None),
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=num_workers,
        drop_last=True,
    )
    return loader, sampler


def save_checkpoint(
    path: str,
    epoch: int,
    model: torch.nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    total_loss: torch.Tensor,
    baseline_loss: torch.Tensor,
    canoncolor_loss: torch.Tensor,
    bbox_loss: torch.Tensor,
) -> None:
    raw_model = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
    payload = {
        "epoch": epoch,
        "model_state_dict": raw_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "total_loss": total_loss.detach(),
        "baseline_loss": baseline_loss.detach(),
        "canoncolor_loss": canoncolor_loss.detach(),
        "bbox_loss": bbox_loss.detach(),
        "lntemperature": raw_model.ln_logit_scale.detach(),
    }
    torch.save(payload, path)


def train_worker(rank: int, world_size: int, args: argparse.Namespace) -> None:
    use_ddp = world_size > 1
    if use_ddp:
        setup_ddp(rank, world_size, args.ddp_port)

    try:
        seed_everything(args.seed, rank)

        if rank == 0:
            os.makedirs(args.ckpt_dir, exist_ok=True)
            print(f"Saving checkpoints to: {args.ckpt_dir}")

        model = PointSemSegWithDecoder(args=args).cuda(rank)
        if args.pretrained_path:
            ckpt = torch.load(args.pretrained_path, map_location=f"cuda:{rank}")
            state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
            model.load_state_dict(state, strict=False)
            if rank == 0:
                print(f"Loaded pretrained weights from {args.pretrained_path}")

        if use_ddp:
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[rank],
                find_unused_parameters=False,
            )

        train_loader, train_sampler = create_train_loader(
            data_root=args.data_root,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            use_ddp=use_ddp,
        )

        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=len(train_loader) * args.n_epoch,
            eta_min=args.eta_min,
        )

        if args.continue_path:
            resume = torch.load(args.continue_path, map_location=f"cuda:{rank}")
            model_state = resume["model_state_dict"] if "model_state_dict" in resume else resume
            (model.module if use_ddp else model).load_state_dict(model_state, strict=False)
            if "optimizer_state_dict" in resume:
                optimizer.load_state_dict(resume["optimizer_state_dict"])
            if "scheduler_state_dict" in resume:
                scheduler.load_state_dict(resume["scheduler_state_dict"])
            if rank == 0:
                print(f"Resumed from {args.continue_path}")

        baseline_criterion = DistillLossContrastive().cuda(rank)
        canoncolor_criterion = CanonicalColorLoss().cuda(rank)
        bbox_criterion = PointBasedBBoxOffsetLoss().cuda(rank)

        global_iter = 0
        for epoch in range(1, args.n_epoch + 1):
            if use_ddp:
                train_sampler.set_epoch(epoch)
            model.train()
            epoch_losses = []

            for batch in train_loader:
                for key, value in batch.items():
                    if isinstance(value, torch.Tensor):
                        batch[key] = value.cuda(rank, non_blocking=True)

                backbone_feat, _, canoncolor_out, _, bbox_pred, _ = model(batch)
                raw_model = model.module if use_ddp else model

                mask_points = batch["mask2pt"]
                mask_embeds = batch["label_embeds"]
                pt_offset = batch["offset"]

                baseline_loss = baseline_criterion(
                    backbone_feat,
                    pt_offset,
                    mask_embeds,
                    mask_points,
                    raw_model.ln_logit_scale,
                )
                canoncolor_loss = canoncolor_criterion(
                    canoncolor_out,
                    batch["canoncial_color"],
                    pt_offset,
                    mask_points,
                )
                bbox_loss = bbox_criterion(
                    bbox_pred=bbox_pred,
                    pts=batch["coord"],
                    pt_offset=pt_offset,
                    mask_points=mask_points,
                )

                weighted_canoncolor = args.canoncolor_loss_weight * canoncolor_loss
                weighted_bbox = args.bbox_loss_weight * bbox_loss
                if epoch > args.n_epoch - args.drop_canoncolor_last_n_epochs:
                    total_loss = baseline_loss + weighted_bbox
                else:
                    total_loss = baseline_loss + weighted_canoncolor + weighted_bbox

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                scheduler.step()

                epoch_losses.append(float(total_loss.item()))
                if rank == 0 and global_iter % args.log_every == 0:
                    temp = float(torch.exp(raw_model.ln_logit_scale).item())
                    print(
                        f"iter={global_iter} epoch={epoch} "
                        f"total={total_loss.item():.4f} "
                        f"baseline={baseline_loss.item():.4f} "
                        f"canon={weighted_canoncolor.item():.4f} "
                        f"bbox={weighted_bbox.item():.4f} "
                        f"lr={scheduler.get_last_lr()[0]:.6f} temp={temp:.2f}"
                    )
                global_iter += 1

            if rank == 0:
                avg_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
                print(f"[Epoch {epoch}/{args.n_epoch}] avg_loss={avg_loss:.4f}")
                if epoch % args.save_every == 0:
                    ckpt_path = os.path.join(args.ckpt_dir, f"ckpt_{epoch}.pth")
                    save_checkpoint(
                        ckpt_path,
                        epoch,
                        model,
                        optimizer,
                        scheduler,
                        total_loss,
                        baseline_loss,
                        canoncolor_loss,
                        bbox_loss,
                    )
                    print(f"Saved checkpoint: {ckpt_path}")

        if rank == 0:
            final_path = os.path.join(args.ckpt_dir, "ckpt_final.pth")
            save_checkpoint(
                final_path,
                args.n_epoch,
                model,
                optimizer,
                scheduler,
                total_loss,
                baseline_loss,
                canoncolor_loss,
                bbox_loss,
            )
            print(f"Saved final checkpoint: {final_path}")
    finally:
        if use_ddp:
            cleanup_ddp()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CoSMo3D release training")
    parser.add_argument("--data_root", required=True, type=str, help="Training dataset root")
    parser.add_argument("--ckpt_dir", default="results/find3d_release", type=str, help="Checkpoint output directory")
    parser.add_argument("--pretrained_path", default=None, type=str, help="Optional pretrained checkpoint")
    parser.add_argument("--continue_path", default=None, type=str, help="Optional resume checkpoint")

    parser.add_argument("--n_epoch", default=200, type=int, help="Total epochs")
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size per GPU")
    parser.add_argument("--lr", default=5e-4, type=float, help="Initial learning rate")
    parser.add_argument("--eta_min", default=5e-5, type=float, help="Cosine scheduler minimum lr")
    parser.add_argument("--num_workers", default=4, type=int, help="DataLoader workers per process")

    parser.add_argument("--canoncolor_loss_weight", default=0.2, type=float, help="Weight for canonical color loss")
    parser.add_argument("--bbox_loss_weight", default=5.0, type=float, help="Weight for bbox loss")
    parser.add_argument(
        "--drop_canoncolor_last_n_epochs",
        default=30,
        type=int,
        help="Disable canonical color loss in the final N epochs",
    )

    parser.add_argument("--save_every", default=5, type=int, help="Checkpoint period (epochs)")
    parser.add_argument("--log_every", default=20, type=int, help="Logging period (iterations)")
    parser.add_argument("--seed", default=123, type=int, help="Random seed")
    parser.add_argument("--ddp_port", default=12345, type=int, help="DDP communication port")

    # Keep these model/data options for compatibility with existing modules.
    parser.add_argument("--use_aug", type=int, default=1, choices=[0, 1], help="Use augmentation in data pipeline")
    parser.add_argument("--normalize_cloud", type=int, default=1, choices=[0, 1], help="Normalize point clouds")
    parser.add_argument("--n_mov_avg", type=int, default=5, help="Moving-average size for dense interpolation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    world_size = torch.cuda.device_count()
    if world_size < 1:
        raise RuntimeError("No CUDA device detected. This training script requires GPU.")

    if world_size > 1:
        mp.spawn(train_worker, args=(world_size, args), nprocs=world_size, join=True)
    else:
        train_worker(rank=0, world_size=1, args=args)


if __name__ == "__main__":
    main()
