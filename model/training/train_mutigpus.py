## Some of the training code builds upon https://github.com/ardianumam/PartDistill/blob/main/train.py
import os
import torch
import argparse
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.optim as optim
from model.backbone.pt3.model import PointSemSeg
from model.data.data import TrainingData, EvalData, collate_fn
from tqdm import tqdm
import numpy as np
from model.evaluation.core import viz_pred_mask, compute_overall_iou_objwise
from model.training.loss import DistillLossContrastive
from transformers import AutoTokenizer, AutoModel
import random


def setup(rank, world_size):
    """初始化分布式环境"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    """清理分布式环境"""
    dist.destroy_process_group()


def create_data_loader(rank, data_root, shuffle_train, shuffle_test, 
                       eval_split, drop_last_train=True, drop_last_test=False, 
                       is_test_only=False, batch_size=16):
    test_data = EvalData(data_root, split=eval_split)
    test_sampler = DistributedSampler(
        test_data, 
        shuffle=shuffle_test
    ) if dist.is_initialized() else None
    
    test_loader = DataLoader(
        test_data, 
        batch_size=1, 
        shuffle=shuffle_test and (test_sampler is None),
        collate_fn=collate_fn, 
        num_workers=0, 
        drop_last=drop_last_test,
        sampler=test_sampler
    )
    
    if is_test_only:    
        return test_loader, test_sampler
    
    train_data = TrainingData(data_root)
    BS = min(batch_size, len(train_data))
    if rank == 0:
        print(f'Batch size per GPU: {BS}')
    
    train_sampler = DistributedSampler(
        train_data, 
        shuffle=shuffle_train
    ) if dist.is_initialized() else None
    
    train_loader = DataLoader(
        train_data, 
        batch_size=BS, 
        shuffle=shuffle_train and (train_sampler is None), 
        collate_fn=collate_fn,
        num_workers=0,
        drop_last=drop_last_train,
        sampler=train_sampler
    )
    
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    total_batch_size = BS * world_size
    train_iter_per_epoch = (len(train_data) // total_batch_size) + 1

    return train_loader, test_loader, train_iter_per_epoch, train_sampler
        

def evaluate(rank, model, dataloader, loss_fn, n_epoch, set_name, 
             eval_loss=True, visualize_idxs=[20,25,55,80,139]):
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    n_visualize_epoch = 5
    prefix = "pt"
    iou_list = []
    loss_list = []
    i = 0
    
    with torch.no_grad():
        iterator = tqdm(dataloader, desc=f"Evaluating {set_name}-set (GPU {rank})") if rank == 0 else dataloader
        
        for data in iterator:
            for key in data.keys():
                if isinstance(data[key], torch.Tensor):
                    data[key] = data[key].cuda(rank, non_blocking=True)

            # 先获取模型输出，再单独获取ln_logit_scale，兼容不支持return_logit_scale的模型
            net_out = model(x=data)
            # 根据是否为DDP模型获取ln_logit_scale
            if world_size > 1:
                ln_logit_scale = model.module.ln_logit_scale
            else:
                ln_logit_scale = model.ln_logit_scale
            temperature = torch.exp(ln_logit_scale)
            
            text_embeds = data['label_embeds']
            masks = data['masks']
            mask_view_idxs = data["mask_view_idxs"]
            point2face = data['point2face']
            pix2face = data['pixel2face']
            labels = data['labels']
            mask_pts = data['mask2pt']
            pt_offset = data['offset']
            
            m = AutoModel.from_pretrained("google/siglip-base-patch16-224").to(rank)
            tokenizer = AutoTokenizer.from_pretrained("google/siglip-base-patch16-224")
            inputs = tokenizer(labels[0], padding="max_length", return_tensors="pt").to(rank)
            
            with torch.no_grad():
                text_feat = m.get_text_features(** inputs)
        
            text_feat = text_feat / (text_feat.norm(dim=-1, keepdim=True) + 1e-12)
            
            iou = compute_overall_iou_objwise(
                pred=net_out,
                text_embeds=text_embeds,
                masks=masks,
                mask_view_idxs=mask_view_idxs,
                point2face=point2face,
                pixel2face=pix2face,
                temperature=temperature
            )
            iou_list.append(iou)

            if eval_loss:
                loss = loss_fn(
                    net_out,
                    pt_offset,
                    text_embeds,
                    mask_pts,
                    ln_logit_scale
                )
                loss_list.append(loss.item())

            if rank == 0 and n_epoch % n_visualize_epoch == 0 and (i in visualize_idxs):
                viz_pred_mask(
                    pred=net_out,
                    text_embeds=text_embeds,
                    texts=[[x] for x in labels[0]],
                    masks=masks,
                    mask_view_idxs=mask_view_idxs,
                    point2face=point2face,
                    pixel2face=pix2face,
                    n_epoch=n_epoch,
                    obj_visualize_idx=i,
                    prefix=f"{prefix}-{set_name}",
                    temperature=temperature
                )
            i += 1
    
    if world_size > 1:
        iou_tensor = torch.tensor(iou_list, device=rank)
        all_iou = [torch.zeros_like(iou_tensor) for _ in range(world_size)]
        dist.all_gather(all_iou, iou_tensor)
        iou_list = [item.item() for sublist in all_iou for item in sublist]
        
        if eval_loss:
            loss_tensor = torch.tensor(loss_list, device=rank)
            all_loss = [torch.zeros_like(loss_tensor) for _ in range(world_size)]
            dist.all_gather(all_loss, loss_tensor)
            loss_list = [item.item() for sublist in all_loss for item in sublist]
    
    miou = np.mean(iou_list) if iou_list else 0.0
    loss = np.mean(loss_list) if loss_list and eval_loss else 0.0
    return miou, loss


def train(rank, world_size, args):
    setup(rank, world_size)
    
    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    random.seed(args.seed + rank)
    
    if rank == 0:
        ckpt_dir = os.path.join(args.ckpt_dir, f"find3d_{args.exp_suffix}")
        os.makedirs(ckpt_dir, exist_ok=True)
    else:
        ckpt_dir = None
    
    model = PointSemSeg(args=args, dim_output=768)
    model = model.to(rank)
    
    # if world_size > 1:
    #     model = torch.nn.parallel.DistributedDataParallel(
    #         model,
    #         device_ids=[rank],
    #         find_unused_parameters=True
    #     )
    #     # 启用静态图模式，解决参数重复标记问题
    #     model._set_static_graph()
    
    if args.pretrained_path:
        if rank == 0:
            print(f"Loading pretrained model from {args.pretrained_path}")
            pretrained_weights = torch.load(args.pretrained_path)
            # 先加载到rank 0的模型
            model.load_state_dict(pretrained_weights["model_state_dict"], strict=False)
            print("Pretrained model loaded on rank 0")
        
        # 关键：将rank 0的参数广播到所有进程
        if world_size > 1:
            for param in model.parameters():
                dist.broadcast(param.data, src=0)  # 从rank 0同步参数到所有进程
            if rank == 0:
                print("Pretrained parameters broadcast to all ranks")
    
    # 最后包装DDP
    if world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[rank],
            find_unused_parameters=False  # 必须关闭冲突配置
        )


    train_loader, test_loader, train_iter_per_epoch, train_sampler = create_data_loader(
        rank, args.data_root,
        shuffle_train=True, shuffle_test=False,
        eval_split="val", drop_last_train=True, 
        drop_last_test=False, batch_size=args.batch_size
    )
    
    train_val_loader, _ = create_data_loader(
        rank, args.data_root,
        shuffle_train=True, shuffle_test=False,
        eval_split="train", is_test_only=True
    )

    opt = optim.Adam(model.parameters(), lr=args.lr)
    
    if args.continue_path:
        if rank == 0:
            print(f"Continuing training from {args.continue_path}")
            checkpoint = torch.load(args.continue_path)
            model.load_state_dict(checkpoint["model_state_dict"])
            opt.load_state_dict(checkpoint["optimizer_state_dict"])
            # 修复参数加载方式
            if world_size > 1:
                model.module.ln_logit_scale = nn.Parameter(checkpoint["lntemperature"].data, requires_grad=True)
            else:
                model.ln_logit_scale = nn.Parameter(checkpoint["lntemperature"].data, requires_grad=True)
            print("Loaded optimizer state")
        
        if world_size > 1:
            for state in opt.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        dist.broadcast(v, src=0)
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        opt, 
        train_iter_per_epoch * args.n_epoch, 
        eta_min=args.eta_min
    )
           
    criterion = DistillLossContrastive()

    global_iter = 0
    model.train()

    for epoch in range(args.n_epoch):
        current_epoch = epoch + 1
        
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        
        loss_epoch_current = []

        train_iterator = tqdm(
            train_loader, 
            desc=f"Training epoch: {current_epoch}/{args.n_epoch} (GPU {rank})"
        ) if rank == 0 else train_loader

        for iter_idx, data in enumerate(train_iterator):
            for key in data.keys():
                if isinstance(data[key], torch.Tensor):
                    data[key] = data[key].to(rank, non_blocking=True)

            mask_points = data['mask2pt'] 
            mask_embeds = data['label_embeds']
            pt_offset = data['offset']
            
            # 先获取模型输出，再单独获取ln_logit_scale，兼容不支持return_logit_scale的模型
            net_out = model(data)
            # 根据是否为DDP模型获取ln_logit_scale
            if world_size > 1:
                ln_logit_scale = model.module.ln_logit_scale
            else:
                ln_logit_scale = model.ln_logit_scale
            
            
            # 使用获取的ln_logit_scale计算损失
            loss = criterion(
                net_out,
                pt_offset,
                mask_embeds,
                mask_points,
                ln_logit_scale
            )
            
            loss_epoch_current.append(loss.item())
            cur_lr = scheduler.get_last_lr()[0]

            opt.zero_grad()
            ############## 有时候backup会报错尝试找原因
            try:  
                loss.backward()  # 可能抛出梯度错误的位置
            except RuntimeError as e:
                if "element 0 of tensors does not require grad" in str(e):
                    # 定义日志文件路径
                    log_dir = "/apdcephfs_cq11/share_303570626/lanejin/project/Find3D/results/logs"
                    log_file = os.path.join(log_dir, "backuperror.txt")
                    
                    # 确保日志目录存在（若不存在则创建）
                    os.makedirs(log_dir, exist_ok=True)
                    
                    # 构建错误信息内容
                    error_msg = [
                        f"\n===== 错误时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} =====",
                        f"Epoch: {current_epoch}, Iter: {iter_idx}, GPU: {rank}",
                        f"错误详情: {str(e)}",
                        f"net_out.requires_grad: {net_out.requires_grad}",
                        f"loss.grad_fn: {loss.grad_fn}",
                        f"模型训练模式: {model.training}",
                        "关键参数梯度状态:"
                    ]
                    # 添加参数信息
                    for name, param in model.named_parameters():
                        if "ln_logit_scale" in name or "backbone" in name:
                            error_msg.append(f"  - {name}: requires_grad={param.requires_grad}")
                    error_msg.append("=========================================\n")
                    
                    # 写入文件（追加模式，避免覆盖历史日志）
                    with open(log_file, "a", encoding="utf-8") as f:
                        f.write("\n".join(error_msg))
                    
                    # 同时打印到控制台（便于实时观察）
                    print("\n".join(error_msg))
                    print("跳过当前迭代，继续训练...\n")
                    continue
                else:
                    raise e
            opt.step()
            scheduler.step()
            
            if rank == 0:
                temperature = np.exp(ln_logit_scale.item())
                print(f"iter {global_iter} loss {loss.item()} lr {cur_lr} temperature {temperature}")
            
            global_iter += 1
        
        if world_size > 1:
            loss_tensor = torch.tensor(np.mean(loss_epoch_current), device=rank)
            all_losses = [torch.zeros_like(loss_tensor) for _ in range(world_size)]
            dist.all_gather(all_losses, loss_tensor)
            epoch_loss_avg = np.mean([loss.item() for loss in all_losses])
        else:
            epoch_loss_avg = np.around(np.mean(loss_epoch_current), decimals=4)

        log_n_epochs = 5
        '''if current_epoch % log_n_epochs == 0:
            miou_test, loss_test = evaluate(
                rank, model.eval(), test_loader, 
                criterion, current_epoch, "val"
            )
            
            miou_train, loss_train_objbatch = evaluate(
                rank, model.eval(), train_val_loader, 
                criterion, current_epoch, "train"
            )
            
            if rank == 0:
                epoch_text_out = f"Epoch {current_epoch}/{args.n_epoch} --> loss: batch-64 train {epoch_loss_avg}  batch-1 train {loss_train_objbatch} val {loss_test}, iou: train {miou_train} val {miou_test}"
                print(epoch_text_out)
            
            model.train()'''
        
        if rank == 0 and current_epoch % 5 == 0:
            ckpt_path_test = os.path.join(ckpt_dir, f"ckpt_{current_epoch}.pth")
            torch.save({
                'epoch': current_epoch,
                'model_state_dict': model.module.state_dict() if world_size > 1 else model.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'loss': loss,
                'lntemperature': model.module.ln_logit_scale if world_size > 1 else model.ln_logit_scale,
                'scheduler_state_dict': scheduler.state_dict()
            }, ckpt_path_test)
    
    if rank == 0:
        final_ckpt_path = os.path.join(ckpt_dir, "ckpt_final.pth")
        torch.save({
            'epoch': args.n_epoch,
            'model_state_dict': model.module.state_dict() if world_size > 1 else model.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            'loss': loss,
            'lntemperature': model.module.ln_logit_scale if world_size > 1 else model.ln_logit_scale,
            'scheduler_state_dict': scheduler.state_dict()
        }, final_ckpt_path)
    
    cleanup()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epoch', type=int, required=True)
    parser.add_argument('--batch_size', default=16, type=int, help="Batch size per GPU")
    parser.add_argument('--lr', type=float, default=0.0003, metavar='LR', help='learning rate')
    parser.add_argument('--eta_min', type=float, default=0.00005, metavar='LR', help='minimum learning rate')
    parser.add_argument('--step', type=int, default=40, help='lr decay step')
    parser.add_argument('--use_aug', type=int, default=1, choices=[0, 1])
    parser.add_argument('--normalize_cloud', type=int, default=1, choices=[0, 1])
    parser.add_argument('--ckpt_dir', type=str, default="checkpoints")
    parser.add_argument('--continue_path', type=str)
    parser.add_argument('--n_mov_avg', type=int, default=5)
    parser.add_argument('--exp_suffix', default='', type=str)
    parser.add_argument('--data_root', required=True, type=str)
    parser.add_argument('--pretrained_path', type=str, default=None, help="Path to pretrained model")

    args = parser.parse_args()
    args.seed = 123
    
    world_size = torch.cuda.device_count()
    print(f"Detected {world_size} available GPU(s)")
    
    if world_size > 1:
        print('----------using DDP---------------')
        mp.spawn(
            train,
            args=(world_size, args),
            nprocs=world_size,
            join=True
        )
    else:
        print('------------single gpu---------------')
        train(0, 1, args)


if __name__ == '__main__':
    main()










'''import os
import torch
import argparse
import numpy as np
import random
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import lightning.pytorch as pl
from lightning.pytorch import seed_everything, Trainer
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from model.backbone.pt3.model import PointSemSeg
from model.data.data import TrainingData, EvalData, collate_fn
from model.evaluation.core import viz_pred_mask, compute_overall_iou_objwise
from model.training.loss import DistillLossContrastive
from transformers import AutoTokenizer, AutoModel


class PointCloudDataModule(pl.LightningDataModule):
    """数据模块，负责数据加载和处理"""
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.data_root = args.data_root
        self.batch_size = args.batch_size
        self.world_size = args.world_size

    def setup(self, stage: str = None):
        # 准备训练和验证数据集
        if stage == 'fit' or stage is None:
            self.train_dataset = TrainingData(self.data_root)
            self.val_dataset = EvalData(self.data_root, split="val")
            self.train_val_dataset = EvalData(self.data_root, split="train")
            
        # 准备测试数据集
        if stage == 'test' or stage is None:
            self.test_dataset = EvalData(self.data_root, split="val")

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        sampler = DistributedSampler(
            self.train_dataset,
            num_replicas=self.world_size,
            rank=self.trainer.local_rank,
            shuffle=True
        ) if self.world_size > 1 else None
        
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=(sampler is None),
            collate_fn=collate_fn,
            num_workers=0,
            drop_last=True,
            pin_memory=False,
            sampler=sampler
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return [
            DataLoader(
                self.val_dataset,
                batch_size=1,
                shuffle=False,
                collate_fn=collate_fn,
                num_workers=0,
                drop_last=False,
                pin_memory=False,
                sampler=DistributedSampler(
                    self.val_dataset,
                    num_replicas=self.world_size,
                    rank=self.trainer.local_rank,
                    shuffle=False
                ) if self.world_size > 1 else None
            ),
            DataLoader(
                self.train_val_dataset,
                batch_size=1,
                shuffle=False,
                collate_fn=collate_fn,
                num_workers=0,
                drop_last=False,
                pin_memory=False,
                sampler=DistributedSampler(
                    self.train_val_dataset,
                    num_replicas=self.world_size,
                    rank=self.trainer.local_rank,
                    shuffle=False
                ) if self.world_size > 1 else None
            )
        ]

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0,
            drop_last=False,
            pin_memory=False
        )


class LitPointSemSeg(pl.LightningModule):
    """Lightning模型类，封装训练逻辑"""
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.save_hyperparameters(args)
        
        # 初始化模型
        self.model = PointSemSeg(args=args, dim_output=768)
        
        # 定义损失函数
        self.criterion = DistillLossContrastive()
        
        # 文本模型和分词器
        self.text_model = AutoModel.from_pretrained("google/siglip-base-patch16-224")
        self.tokenizer = AutoTokenizer.from_pretrained("google/siglip-base-patch16-224")
        
        # 可视化参数
        self.n_visualize_epoch = 5
        self.visualize_idxs = [20, 25, 55, 80, 139]
        self.prefix = "pt"

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        # 提取数据
        mask_points = batch['mask2pt']
        mask_embeds = batch['label_embeds']
        pt_offset = batch['offset']
        
        # 模型前向传播
        net_out = self.model(batch)
        
        # 计算损失
        loss = self.criterion(
            net_out,
            pt_offset,
            mask_embeds,
            mask_points,
            self.model.ln_logit_scale
        )
        
        # 记录训练损失
        self.log('train/loss', loss, sync_dist=True, prog_bar=True)
        
        # 打印训练信息
        if self.global_rank == 0 and batch_idx % 10 == 0:
            current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
            temperature = np.exp(self.model.ln_logit_scale.item())
            self.log('train/lr', current_lr, sync_dist=True)
            self.log('train/temperature', temperature, sync_dist=True)
            self.print(f"Iter {self.global_step} | Loss: {loss.item():.4f} | LR: {current_lr:.6f} | Temp: {temperature:.2f}")
        
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        # 区分验证集和训练集评估
        set_name = "val" if dataloader_idx == 0 else "train"
        
        # 提取数据
        data = batch
        for key in data.keys():
            if isinstance(data[key], torch.Tensor):
                data[key] = data[key].to(self.device)
        
        # 模型前向传播
        net_out = self.model(x=data)
        
        # 提取必要数据
        text_embeds = data['label_embeds']
        masks = data['masks']
        mask_view_idxs = data["mask_view_idxs"]
        point2face = data['point2face']
        pix2face = data['pixel2face']
        labels = data['labels']
        mask_pts = data['mask2pt']
        pt_offset = data['offset']
        
        # 处理文本特征
        inputs = self.tokenizer(labels[0], padding="max_length", return_tensors="pt").to(self.device)
        with torch.no_grad():
            text_feat = self.text_model.get_text_features(** inputs)
        
        # 归一化文本特征
        text_feat = text_feat / (text_feat.norm(dim=-1, keepdim=True) + 1e-12)
        
        # 计算温度参数
        temperature = torch.exp(self.model.ln_logit_scale)
        
        # 计算IOU
        iou = compute_overall_iou_objwise(
            pred=net_out,
            text_embeds=text_embeds,
            masks=masks,
            mask_view_idxs=mask_view_idxs,
            point2face=point2face,
            pixel2face=pix2face,
            temperature=temperature
        )
        
        # 计算损失
        loss = self.criterion(
            net_out,
            pt_offset,
            text_embeds,
            mask_pts,
            self.model.ln_logit_scale
        )
        
        # 记录指标
        self.log(f'{set_name}/iou', iou, sync_dist=True, add_dataloader_idx=False)
        self.log(f'{set_name}/loss', loss, sync_dist=True, add_dataloader_idx=False)
        
        # 可视化（只在主进程和特定epoch）
        current_epoch = self.current_epoch
        if (self.global_rank == 0 and 
            current_epoch % self.n_visualize_epoch == 0 and 
            batch_idx in self.visualize_idxs):
            
            viz_pred_mask(
                pred=net_out,
                text_embeds=text_embeds,
                texts=[[x] for x in labels[0]],
                masks=masks,
                mask_view_idxs=mask_view_idxs,
                point2face=point2face,
                pixel2face=pix2face,
                n_epoch=current_epoch,
                obj_visualize_idx=batch_idx,
                prefix=f"{self.prefix}-{set_name}",
                temperature=temperature
            )
        
        return {'iou': iou, 'loss': loss.item()}

    def on_validation_epoch_end(self):
        # 计算并打印 epoch 级别的指标
        if self.global_rank == 0:
            train_iou = self.trainer.callback_metrics.get('train/iou', 0)
            train_loss = self.trainer.callback_metrics.get('train/loss', 0)
            val_iou = self.trainer.callback_metrics.get('val/iou', 0)
            val_loss = self.trainer.callback_metrics.get('val/loss', 0)
            
            self.print(f"\nEpoch {self.current_epoch} Summary:")
            self.print(f"Train Loss: {train_loss:.4f} | Train IoU: {train_iou:.4f}")
            self.print(f"Val Loss: {val_loss:.4f} | Val IoU: {val_iou:.4f}\n")

    def configure_optimizers(self):
        # 配置优化器和学习率调度器
        optimizer = optim.Adam(self.parameters(), lr=self.args.lr)
        train_iter_per_epoch = len(self.trainer.datamodule.train_dataloader())
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=train_iter_per_epoch * self.args.n_epoch,
            eta_min=self.args.eta_min
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step"
            }
        }

    def on_load_checkpoint(self, checkpoint):
        # 加载检查点时恢复温度参数
        if 'lntemperature' in checkpoint:
            self.model.ln_logit_scale = checkpoint['lntemperature']


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epoch', type=int, required=True, help="Number of training epochs")
    parser.add_argument('--batch_size', default=16, type=int, help="Batch size per GPU")
    parser.add_argument('--lr', type=float, default=0.0003, help='Initial learning rate')
    parser.add_argument('--eta_min', type=float, default=0.00005, help='Minimum learning rate')
    parser.add_argument('--use_aug', type=int, default=1, choices=[0, 1], help="Use data augmentation")
    parser.add_argument('--normalize_cloud', type=int, default=1, choices=[0, 1], help="Normalize point cloud")
    parser.add_argument('--ckpt_dir', type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument('--continue_path', type=str, default=None, help="Path to checkpoint to continue training")
    parser.add_argument('--exp_suffix', default='', type=str, help="Suffix for experiment directory")
    parser.add_argument('--data_root', required=True, type=str, help="Root directory of dataset")
    parser.add_argument('--pretrained_path', type=str, default=None, help="Path to pretrained model weights")
    parser.add_argument('--seed', type=int, default=123, help="Random seed for reproducibility")
    parser.add_argument('--precision', type=str, default='16-mixed', help="Precision mode (e.g., '16-mixed', '32-true')")
    
    args = parser.parse_args()
    
    # 设置随机种子
    seed_everything(args.seed, workers=True)
    
    # 检测可用GPU数量
    args.world_size = torch.cuda.device_count()
    print(f"Detected {args.world_size} available GPU(s)")
    
    # 创建检查点目录
    ckpt_dir = os.path.join(args.ckpt_dir, f"find3d_{args.exp_suffix}")
    os.makedirs(ckpt_dir, exist_ok=True)
    
    # 配置检查点回调
    checkpoint_callback = ModelCheckpoint(
        monitor="val/iou",
        dirpath=ckpt_dir,
        filename="{epoch:02d}-{val/iou:.4f}",
        save_top_k=5,
        save_last=True,
        every_n_epochs=5,
        mode="max",
        verbose=True
    )
    
    # 初始化数据模块
    dm = PointCloudDataModule(args)
    
    # 初始化模型
    model = LitPointSemSeg(args)
    
    # 加载预训练模型
    if args.pretrained_path:
        print(f"Loading pretrained model from {args.pretrained_path}")
        pretrained_weights = torch.load(args.pretrained_path)
        model.load_state_dict(pretrained_weights["model_state_dict"], strict=False)
    
    # 配置Trainer
    trainer = Trainer(
        devices=-1,  # 使用所有可用GPU
        accelerator="gpu",
        precision=args.precision,
        strategy=DDPStrategy(find_unused_parameters=True),
        max_epochs=args.n_epoch,
        log_every_n_steps=1,
        callbacks=[checkpoint_callback],
        num_sanity_val_steps=0,  # 禁用验证集的sanity check
        enable_progress_bar=True,
        enable_model_summary=True
    )
    
    # 开始训练
    trainer.fit(
        model,
        datamodule=dm,
        ckpt_path=args.continue_path  # 继续训练的检查点路径
    )


if __name__ == '__main__':
    main()'''
