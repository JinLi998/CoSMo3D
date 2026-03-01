import torch.nn as nn
import torch.nn.functional as F
import torch
from sklearn.metrics.pairwise import cosine_similarity

# note that batching happens with offsets - all point clouds are concatenated and the offsets
# denote the start/end of each object





#########debug 用
##找到特征最像的点
def find_similar_points(feats, n):
    """
    找到每个点的特征最相似的n个点的索引
    
    参数:
        feats: (N, D) 数组，N个点的特征，每个特征维度为D
        n: 要查找的最相似点的数量（不包含自身）
    
    返回:
        similar_indices: (N, n) 数组，第i行是第i个点的n个最相似点的索引
    """
    N = feats.shape[0]
    if n >= N:
        raise ValueError("n不能大于等于点的总数")
    
    # 计算特征间的余弦相似度（值越大越相似）
    similarity = cosine_similarity(feats)  # 形状为 (N, N)，similarity[i][j] 是点i和点j的相似度
    
    # 对每个点，找到相似度最高的n个点（排除自身）
    similar_indices = np.zeros((N, n), dtype=int)
    for i in range(N):
        # 对第i个点的相似度排序，返回索引（从大到小）
        sorted_idx = np.argsort(similarity[i])[::-1]  # [::-1] 逆序为从大到小
        # 排除自身（sorted_idx[0]是i本身），取前n个
        similar_indices[i] = sorted_idx[1 : 1 + n]
    
    return similar_indices
# 特征最像点的可视化
def save_similar_points_as_ply(pc, similar_indices, target_idx, save_path, n_colors=None):
    """
    将目标点、其相似点及其他点保存为PLY点云（不同颜色标记）
    
    参数:
        pc: (N, 3) 点云坐标（x, y, z）
        similar_indices: (N, n) 相似点索引
        target_idx: 目标点索引
        save_path: PLY文件保存路径（超参数，如"similar_points.ply"）
        n_colors: 自定义颜色列表，格式为 [(r,g,b), ...]，长度为n+1（目标点+相似点）
                  默认为：目标点(红)，相似点(蓝)，其他点(灰)
    """
    N = pc.shape[0]
    n = similar_indices.shape[1]
    
    # 1. 定义颜色（RGB值范围0-255）
    if n_colors is None:
        target_color = (255, 0, 0)       # 目标点：红色
        similar_color = (0, 0, 255)      # 相似点：蓝色
        other_color = (128, 128, 128)    # 其他点：灰色
    else:
        if len(n_colors) != n + 1:
            raise ValueError("n_colors长度必须为n+1（目标点+%d个相似点）" % n)
        target_color = n_colors[0]
        similar_color = n_colors[1:]
    
    # 2. 标记每个点的颜色
    colors = np.zeros((N, 3), dtype=np.uint8)  # 初始化所有点为灰色
    colors[:] = other_color
    
    # 标记目标点
    colors[target_idx] = target_color
    
    # 标记相似点
    similar_ids = similar_indices[target_idx]
    colors[similar_ids] = similar_color if n == 1 else np.tile(similar_color, (n, 1))  # 处理n>1的情况
    
    # 3. 写入PLY文件
    with open(save_path, 'w') as f:
        # PLY文件头
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write("element vertex %d\n" % N)
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        
        # 写入点坐标和颜色
        for i in range(N):
            x, y, z = pc[i]
            r, g, b = colors[i]
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b}\n")
    
    print(f"PLY点云已保存至: {save_path}")

################## 对于标签正确的数据集，不用对比损失，直接使用特征进行监督，并用最优传输，理论上会更好
class DistillLossSimpleMSE(nn.Module):
    def __init__(self):
        super(DistillLossSimpleMSE, self).__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, net_out, pt_offset, mask_embs, mask_pts, logit_scale, is_backward_dist=False, stud_knowledge=None, partfieldfeats=None):
        """
        最简单的MSE特征监督，严格匹配输入参数格式
        """
        total_loss = 0.0
        total_points = 0  # 统计参与计算的总点数

        # 设备一致性处理
        device = net_out.device
        obj_start_idxs = torch.cat((torch.tensor([0], device=device), pt_offset[:-1].to(device)))

        for obj_idx in range(len(pt_offset) - 1):
            # 获取当前物体的点特征
            start_idx = obj_start_idxs[obj_idx]
            end_idx = pt_offset[obj_idx]
            obj_pts_feats = net_out[start_idx:end_idx, :].to(device)

            # 获取当前物体的mask
            obj_masks = mask_pts[obj_idx].to(device)
            n_cur_masks = obj_masks.shape[0]

            # 遍历每个mask
            for mask_idx in range(n_cur_masks):
                mask = obj_masks[mask_idx]
                mask_sum = torch.sum(mask)
                if mask_sum == 0:
                    continue

                # 提取mask内点特征
                masked_pts = obj_pts_feats[mask.bool()]
                n_pts = masked_pts.shape[0]
                if n_pts == 0:
                    continue

                # 获取对应标签特征（全局索引计算）
                global_mask_idx = sum(len(m) for m in mask_pts[:obj_idx]) + mask_idx
                label_emb = mask_embs[global_mask_idx].to(device)
                label_emb = torch.nan_to_num(label_emb, nan=0.0, posinf=0.0, neginf=0.0)

                # 扩展标签特征并计算MSE
                label_expanded = label_emb.unsqueeze(0).repeat(n_pts, 1)
                loss = self.mse_loss(masked_pts, label_expanded)
                
                # 累加损失（按点数加权）
                total_loss += loss * n_pts
                total_points += n_pts

        # 避免除以零
        if total_points == 0:
            return torch.tensor(0.0, device=device)
        return total_loss / total_points
        
class SinkhornDistance(nn.Module):
    """优化后的Sinkhorn算法，解决数值不稳定导致的NaN问题"""
    def __init__(self, eps=1e-2, max_iter=50, reduction='none', clip_min=1e-8):
        super(SinkhornDistance, self).__init__()
        self.eps = eps  # 增大正则化系数，减少指数溢出风险（原1e-3→1e-2）
        self.max_iter = max_iter  # 减少迭代次数，降低累积误差（原100→50）
        self.reduction = reduction
        self.clip_min = clip_min  # 防止log(0)的最小值

    def forward(self, x, y):
        """
        x: [batch_size, n_points, dim] 点特征分布
        y: [batch_size, m_points, dim] 目标分布
        返回: 两个分布的Wasserstein距离（数值稳定版）
        """
        # 计算成本矩阵（L2距离，添加epsilon防止为0）
        x_flat = x.view(x.size(0), x.size(1), 1, x.size(2))
        y_flat = y.view(y.size(0), 1, y.size(1), y.size(2))
        C = torch.sum((x_flat - y_flat) **2, dim=-1)  # [batch_size, n_points, m_points]
        C = C.clamp(min=self.clip_min)  # 确保成本矩阵无0，避免后续log(0)

        # 初始化边际分布（均匀分布，添加epsilon防止除以0）
        n = x.size(1)
        m = y.size(1)
        mu = (torch.ones(x.size(0), n, 1, device=x.device) / (n + self.clip_min)).clamp(min=self.clip_min)
        nu = (torch.ones(y.size(0), 1, m, device=x.device) / (m + self.clip_min)).clamp(min=self.clip_min)

        # 初始化u和v为正值，避免后续exp(-inf)导致的0
        u = torch.full_like(mu, fill_value=1.0 / n, device=x.device).clamp(min=self.clip_min)
        v = torch.full_like(nu, fill_value=1.0 / m, device=x.device).clamp(min=self.clip_min)

        # Sinkhorn迭代（添加数值稳定处理）
        for _ in range(self.max_iter):
            # 更新v：避免log(0)和exp溢出
            sum_u = torch.sum(u * torch.exp(-C / self.eps), dim=1, keepdim=True).clamp(min=self.clip_min)
            log_sum_u = torch.log(sum_u)
            v = torch.exp(-self.eps * (log_sum_u + nu / self.eps)).clamp(min=self.clip_min)

            # 更新u：对称处理
            sum_v = torch.sum(v * torch.exp(-C / self.eps), dim=2, keepdim=True).clamp(min=self.clip_min)
            log_sum_v = torch.log(sum_v)
            u = torch.exp(-self.eps * (log_sum_v + mu / self.eps)).clamp(min=self.clip_min)

        # 计算最优传输矩阵和Wasserstein距离（最终校验）
        T = u * torch.exp(-C / self.eps) * v
        distance = torch.sum(T * C, dim=(1, 2)).clamp(min=self.clip_min, max=1e6)  # 限制距离范围，避免极端值

        # 处理可能的NaN/Inf（兜底方案）
        distance = torch.nan_to_num(distance, nan=0.0, posinf=1e6, neginf=0.0)

        if self.reduction == 'mean':
            return distance.mean()
        elif self.reduction == 'sum':
            return distance.sum()
        return distance

class DistillLossOT(nn.Module):
    def __init__(self, eps=1e-2, max_iter=50, clip_min=1e-8):
        super(DistillLossOT, self).__init__()
        self.ot_loss = SinkhornDistance(
            eps=eps, 
            max_iter=max_iter, 
            reduction='mean', 
            clip_min=clip_min
        )

    def forward(self, net_out, pt_offset, mask_embs, mask_pts, is_backward_dist=False, stud_knowledge=None, partfieldfeats=None):
        total_loss = 0.0
        count = 0

        # 确保所有张量在同一设备（防止混合CPU/GPU导致的误差）
        device = net_out.device
        obj_start_idxs = torch.cat((torch.tensor([0], device=device), pt_offset[:-1].to(device)))

        for obj_idx in range(len(pt_offset) - 1):
            start_idx = obj_start_idxs[obj_idx]
            end_idx = pt_offset[obj_idx]
            obj_pts_feats = net_out[start_idx:end_idx, :]  # [n_pts_cur_obj, dim_ft]

            obj_masks = mask_pts[obj_idx].to(device)  # 确保mask在同一设备
            n_cur_masks = obj_masks.shape[0]

            for mask_idx in range(n_cur_masks):
                mask = obj_masks[mask_idx]
                mask_sum = torch.sum(mask)
                if mask_sum == 0:
                    continue

                # 提取mask内点特征（过滤点数过少的mask，避免分布不稳定）
                if mask_sum < 3:  # 至少3个点才能形成有效分布
                    continue

                masked_pts_feats = obj_pts_feats[mask.bool()]
                pt_dist = masked_pts_feats.unsqueeze(0)  # [1, n_pts_in_mask, dim_ft]

                # 标签特征处理（确保无NaN）
                label_emb = mask_embs[count].to(device)
                label_emb = torch.nan_to_num(label_emb, nan=0.0)  # 清除标签特征中的NaN
                label_dist = label_emb.unsqueeze(0).unsqueeze(0)  # [1, 1, dim_ft]

                # 计算OT损失（再次兜底处理）
                try:
                    ot_distance = self.ot_loss(pt_dist, label_dist)
                    if not torch.isfinite(ot_distance):
                        continue  # 跳过无效损失
                    total_loss += ot_distance
                    count += 1
                except Exception as e:
                    print(f"计算OT损失时出错: {e}")
                    continue

        if count == 0:
            return torch.tensor(0.0, device=device)
        return total_loss / count

######### 对于数据有噪声，使用对比损失会更好
class DistillLossContrastive(nn.Module):
    def __init__(self):
        super(DistillLossContrastive, self).__init__()

    def forward(self, net_out, pt_offset, mask_embs, mask_pts, logit_scale, is_backward_dist=False, stud_kowledge=None, partfieldfeats=None): 

        """
        net_out: [n_total_pts, dim_ft]
        partfieldfeats : [n_total_pts, dim_pf]
        pt_offset: [BS+1,] # start of each object's point idx, the last one is value n_total_pts we can take the [:BS], e.g. [ 3984,   8430,  12707,  16621] if total 16621 pts
        mask_offset: [BS+1,] # start of each object's mask idx, the last one is value n_total_masks we can take the [:BS]
        mask_embs: [n_totak_masks, dim_ft] n_masks_max max number of masks for objects in this batch, padded with 0
        mask_pts: list of [n_cur_masks, n_pts_cur_obj] binary, each entry indicates whether the point is visible for given object's given mask 
        """

        N_MASKS_TOTAL, N_DIM = mask_embs.shape
        mask_npts = [torch.sum(mask_pt, dim=1).view(-1,1) for mask_pt in mask_pts] # list of size(n_cur_mask,0), each is number of points for given mask
        all_masks_npts = torch.cat(mask_npts).cuda()
        mask_nopts = ((all_masks_npts==0)*1).squeeze()
        # since n_pts is different per object, this cannot be vectorized so has to be done sequentially, sadly
        obj_start_idxs = torch.cat((torch.tensor([0]).cuda(), pt_offset[:-1])) # e.g. [0,25,300] whereas pt_offset is [25,300,450]

        sum_feats = [mask_pt.cuda()*1.0 @ net_out[start_idx:end_idx,:] for (mask_pt, start_idx, end_idx) in zip(mask_pts, obj_start_idxs, pt_offset)] # each item is n_cur_masks, n_pts_cur, times n_pts_cur, n_dim, resulting in n_cur_masks, n_dim
        all_sum_feats = torch.cat(sum_feats)
        all_mask_avg_feats = all_sum_feats / (all_masks_npts+1e-12) # n_total_masks, out_dim

        # get dot product with text
        logits = mask_embs @ all_mask_avg_feats.T * torch.exp(logit_scale)  # * torch.exp(logit_scale) # this is (n_total_mask, n_total_mask), row=text col=point  logit_scale 是温度系数，值越大对区分正负样本更敏感
        # print('mask_embs @ all_mask_avg_feats.T shape: ', mask_embs @ all_mask_avg_feats.T , (mask_embs @ all_mask_avg_feats.T ).shape)
        target = torch.arange(N_MASKS_TOTAL).cuda() # size (BS*N_MASKS_MAX,) this is label for diagonal
        modified_target = mask_nopts * -100 + (1-mask_nopts)*target # if no point, -100, otherwise, embedding
        texts_loss = F.cross_entropy(logits, modified_target, reduction='none') # BS*N_MASKS_MAX - CE across texts
        pts_loss = F.cross_entropy(logits.T, modified_target, reduction='none') # BS*N_MASKS_MAX - CE across images
        
        # disregard zeros
        if texts_loss.sum()>0:
            texts_loss_nonzero_avg = (texts_loss[texts_loss>0]).mean()
        else:
            texts_loss_nonzero_avg = torch.tensor(0)

        if pts_loss.sum()>0:
            pts_loss_nonzero_avg = (pts_loss[pts_loss>0]).mean()
        else:
            pts_loss_nonzero_avg = torch.tensor(0)
        
        # print('texts_loss_nonzero_avg: ', texts_loss_nonzero_avg, 'pts_loss_nonzero_avg: ', pts_loss_nonzero_avg)
        loss =  (texts_loss_nonzero_avg + pts_loss_nonzero_avg) / 2.0

        # if is_backward_dist:
        #     pass

        return loss


################ 添加损失，partfield 特征相似且属于相同part的作为正样本点 ； 添加点坐标，将距离锚定点最近的不同part label的点作为负样本点
import torch
import torch.nn as nn
import torch.nn.functional as F


class FieldDistillLossContrastive(nn.Module):
    def __init__(self, topk=10, margin=0.3, part_weight=1.0, min_valid=3):
        super(FieldDistillLossContrastive, self).__init__()
        self.topk = topk  # 正负样本数量
        self.margin = margin  # 对比损失边界
        self.part_weight = part_weight  # 部件约束权重
        self.min_valid = min_valid  # 最小有效样本数
        self.eps = 1e-12  # 数值稳定性常数

    def forward(self, net_out, pt_offset, mask_embs, mask_pts, logit_scale, 
                is_backward_dist=False, stud_knowledge=None, partfieldfeats=None, pc_coor=None): 

        # --------------------------
        # 原始蒸馏损失计算（保持不变）
        # --------------------------
        N_MASKS_TOTAL, N_DIM = mask_embs.shape
        mask_npts = [torch.sum(mask_pt, dim=1).view(-1,1) for mask_pt in mask_pts]
        all_masks_npts = torch.cat(mask_npts).cuda()
        mask_nopts = ((all_masks_npts==0)*1).squeeze()

        obj_start_idxs = torch.cat((torch.tensor([0]).cuda(), pt_offset[:-1]))
        sum_feats = [mask_pt.cuda()*1.0 @ net_out[start_idx:end_idx,:] 
                    for (mask_pt, start_idx, end_idx) in zip(mask_pts, obj_start_idxs, pt_offset)]
        all_sum_feats = torch.cat(sum_feats)
        all_mask_avg_feats = all_sum_feats / (all_masks_npts + self.eps)

        logits = mask_embs @ all_mask_avg_feats.T * torch.exp(logit_scale)
        target = torch.arange(N_MASKS_TOTAL).cuda()
        modified_target = mask_nopts * -100 + (1 - mask_nopts) * target

        texts_loss = F.cross_entropy(logits, modified_target, reduction='none')
        pts_loss = F.cross_entropy(logits.T, modified_target, reduction='none')

        texts_loss_nonzero_avg = texts_loss[texts_loss > 0].mean() if texts_loss.sum() > 0 else torch.tensor(0.0, device=net_out.device)
        pts_loss_nonzero_avg = pts_loss[pts_loss > 0].mean() if pts_loss.sum() > 0 else torch.tensor(0.0, device=net_out.device)

        # --------------------------
        # 优化部分：结合空间距离和语义label的特征约束损失
        # 核心优化：向量化计算 + 减少循环嵌套 + 预计算掩码
        # --------------------------
        part_loss = torch.tensor(0.0, device=net_out.device)
        if partfieldfeats is not None and pc_coor is not None:
            total_valid_pts = 0  # 统计有效计算的点数量，用于归一化
            
            # 遍历每个物体（保留这层循环，因为不同物体需独立处理）
            for obj_idx in range(len(pt_offset) - 1):
                start_idx = obj_start_idxs[obj_idx]
                end_idx = pt_offset[obj_idx]
                obj_pts_count = end_idx - start_idx
                if obj_pts_count < self.topk + 1:
                    continue

                # 1. 提取物体级数据（一次性提取所有需要的特征）
                obj_pf = partfieldfeats[start_idx:end_idx]  # [N, dim_pf]
                obj_net = net_out[start_idx:end_idx]        # [N, dim_ft]
                obj_coor = pc_coor[start_idx:end_idx]       # [N, 3]
                obj_mask = mask_pts[obj_idx].cuda()         # [C, N]，C为类别数
                N, C = obj_pts_count, obj_mask.shape[0]

                # 2. 预计算点的语义label（向量化实现）
                # point_label: [N]，每个点的类别索引（-1表示无类别）
                point_label = torch.full((N,), -1, device=net_out.device, dtype=torch.long)
                label_masks = obj_mask.transpose(0, 1).bool()  # [N, C]，转置后便于并行处理
                for c in range(C):
                    point_label[label_masks[:, c]] = c  # 批量赋值类别

                # 过滤无类别点
                valid_pts_mask = (point_label != -1)
                valid_pts = torch.where(valid_pts_mask)[0]
                if len(valid_pts) == 0:
                    continue
                N_valid = len(valid_pts)

                # 3. 预计算所有相似度矩阵（一次性计算整个物体的矩阵）
                obj_pf_norm = F.normalize(obj_pf, dim=1)
                obj_net_norm = F.normalize(obj_net, dim=1)
                pf_sim = torch.matmul(obj_pf_norm, obj_pf_norm.T)  # [N, N]
                pf_sim.fill_diagonal_(-1.0)  # 批量排除自身
                net_sim = torch.matmul(obj_net_norm, obj_net_norm.T)  # [N, N]

                # 4. 向量化筛选正负样本（核心加速部分）
                # 4.1 生成所有有效点的类别掩码
                anchor_labels = point_label[valid_pts]  # [N_valid]
                same_label_mask = (point_label.unsqueeze(0) == anchor_labels.unsqueeze(1))  # [N_valid, N]
                same_label_mask[:, valid_pts] &= (torch.arange(N, device=net_out.device) != valid_pts.unsqueeze(1))  # 排除自身
                
                # 4.2 批量计算正样本
                # 提取同类别点的相似度并排序
                same_sim = pf_sim[valid_pts] * same_label_mask  # [N_valid, N]，不同类别的点相似度置0
                topk_sim, topk_indices = torch.topk(same_sim, self.topk, dim=1)  # [N_valid, K]
                
                # 过滤有效正样本
                pos_valid_mask = (topk_sim > 0.5).sum(dim=1) >= self.min_valid  # [N_valid]
                valid_anchors = valid_pts[pos_valid_mask]  # 有效锚点索引
                if len(valid_anchors) == 0:
                    continue

                # 4.3 批量计算负样本（基于空间距离）
                # 提取有效锚点的坐标
                anchor_coors = obj_coor[valid_anchors]  # [M, 3]，M为有效锚点数
                # 计算与所有点的距离（向量化）
                dists = torch.norm(anchor_coors.unsqueeze(1) - obj_coor.unsqueeze(0), dim=2)  # [M, N]
                
                # 只保留不同类别的点
                diff_label_mask = (point_label.unsqueeze(0) != anchor_labels[pos_valid_mask].unsqueeze(1))  # [M, N]
                diff_label_mask &= (point_label.unsqueeze(0) != -1)  # 排除无类别点
                dists = dists * diff_label_mask.float() + (1 - diff_label_mask.float()) * 1e18  # 无效点距离设为极大值
                
                # 取最近的topk点
                topk_dists, topk_neg_indices = torch.topk(dists, self.topk, dim=1, largest=False)  # [M, K]
                
                # 过滤有效负样本（距离小于均值）
                dist_mean = dists[dists < 1e18].mean() if (dists < 1e18).any() else 1e18
                neg_valid_mask = (topk_dists < dist_mean).sum(dim=1) >= self.min_valid  # [M]
                final_valid_mask = neg_valid_mask  # 最终有效锚点掩码
                if not final_valid_mask.any():
                    continue

                # 4.4 批量计算损失
                M_valid = final_valid_mask.sum().item()
                if M_valid == 0:
                    continue
                
                # 提取有效正样本和负样本的相似度
                valid_topk_indices = topk_indices[pos_valid_mask][final_valid_mask]  # [M_valid, K]
                valid_topk_neg_indices = topk_neg_indices[final_valid_mask]  # [M_valid, K]
                
                # 计算平均相似度
                pos_sim = torch.gather(net_sim[valid_anchors[final_valid_mask]], 1, valid_topk_indices).mean(dim=1)  # [M_valid]
                neg_sim = torch.gather(net_sim[valid_anchors[final_valid_mask]], 1, valid_topk_neg_indices).mean(dim=1)  # [M_valid]
                
                # 累加损失
                part_loss += F.relu(self.margin - (pos_sim - neg_sim)).sum()
                total_valid_pts += M_valid

            # 归一化损失（按有效锚点数平均）
            if total_valid_pts > 0:
                part_loss = part_loss / total_valid_pts

        # 总损失
        total_loss = (texts_loss_nonzero_avg + pts_loss_nonzero_avg) / 2.0 + self.part_weight * part_loss
        return total_loss


'''################ 添加损失，partfield 特征相似且属于相同part的作为正样本点 ； 属于不同part 且特征非常不相似作为负样本点  （有问题，这样负样本点本身就离的很远，起不到约束作用）
import torch
import torch.nn as nn
import torch.nn.functional as F

class FieldDistillLossContrastive(nn.Module):
    def __init__(self, topk=10, margin=0.3, part_weight=1.0):
        super(FieldDistillLossContrastive, self).__init__()
        self.topk = topk  # 每个点选取的同part最相似点数量
        self.margin = margin  # 对比损失的边界值
        self.part_weight = part_weight  # 部件约束损失的权重

    def forward(self, net_out, pt_offset, mask_embs, mask_pts, logit_scale, 
                is_backward_dist=False, stud_knowledge=None, partfieldfeats=None): 

        """
        net_out: [n_total_pts, dim_ft]
        partfieldfeats : [n_total_pts, dim_pf]（此处仅用于提取点，实际part由mask_pts确定）
        pt_offset: [BS+1,] 每个物体的点起始索引
        mask_embs: [n_total_masks, dim_ft] 掩码嵌入
        mask_pts: 列表，每个元素为 [n_cur_masks, n_pts_cur_obj] 二进制掩码（n_cur_masks为该物体的part数量）
        """

        # --------------------------
        # 原始蒸馏损失计算（保持不变）
        # --------------------------
        N_MASKS_TOTAL, N_DIM = mask_embs.shape
        mask_npts = [torch.sum(mask_pt, dim=1).view(-1,1) for mask_pt in mask_pts]
        all_masks_npts = torch.cat(mask_npts).cuda()
        mask_nopts = ((all_masks_npts==0)*1).squeeze()

        obj_start_idxs = torch.cat((torch.tensor([0]).cuda(), pt_offset[:-1]))
        sum_feats = [mask_pt.cuda()*1.0 @ net_out[start_idx:end_idx,:] 
                    for (mask_pt, start_idx, end_idx) in zip(mask_pts, obj_start_idxs, pt_offset)]
        all_sum_feats = torch.cat(sum_feats)
        all_mask_avg_feats = all_sum_feats / (all_masks_npts + 1e-12)

        logits = mask_embs @ all_mask_avg_feats.T * torch.exp(logit_scale)
        target = torch.arange(N_MASKS_TOTAL).cuda()
        modified_target = mask_nopts * -100 + (1 - mask_nopts) * target

        texts_loss = F.cross_entropy(logits, modified_target, reduction='none')
        pts_loss = F.cross_entropy(logits.T, modified_target, reduction='none')

        texts_loss_nonzero_avg = texts_loss[texts_loss > 0].mean() if texts_loss.sum() > 0 else torch.tensor(0.0, device=net_out.device)
        pts_loss_nonzero_avg = pts_loss[pts_loss > 0].mean() if pts_loss.sum() > 0 else torch.tensor(0.0, device=net_out.device)

        # --------------------------
        # 新增：结合语义label的特征约束损失
        # 核心：正样本同label+高相似度，负样本不同label+低相似度
        # --------------------------
        part_loss = torch.tensor(0.0, device=net_out.device)
        if partfieldfeats is not None:
            # 遍历每个物体
            for obj_idx in range(len(pt_offset) - 1):
                start_idx = obj_start_idxs[obj_idx]  # 物体点全局起始索引
                end_idx = pt_offset[obj_idx]         # 物体点全局结束索引
                obj_pts_count = end_idx - start_idx  # 物体内点数
                if obj_pts_count < self.topk + 1:
                    continue

                # 1. 提取物体级数据
                obj_pf = partfieldfeats[start_idx:end_idx]  # [n_obj_pts, dim_pf]
                obj_pf_norm = F.normalize(obj_pf, dim=1)
                obj_net = net_out[start_idx:end_idx]        # [n_obj_pts, dim_ft]
                obj_net_norm = F.normalize(obj_net, dim=1)
                obj_mask = mask_pts[obj_idx].cuda()         # [n_labels, n_obj_pts] 语义掩码
                n_labels = obj_mask.shape[0]

                # 2. 预计算点的语义label（每个点属于哪个label）
                # 生成点到label的映射：point_label[i] = 点i所属的label索引（-1表示无label）
                point_label = torch.full((obj_pts_count,), -1, device=net_out.device, dtype=torch.long)
                for label_idx in range(n_labels):
                    label_pts = torch.where(obj_mask[label_idx] == 1)[0]
                    point_label[label_pts] = label_idx

                # 3. 计算相似度矩阵
                pf_sim = torch.matmul(obj_pf_norm, obj_pf_norm.T)  # [n_obj_pts, n_obj_pts]
                net_sim = torch.matmul(obj_net_norm, obj_net_norm.T)  # [n_obj_pts, n_obj_pts]

                # 4. 对每个点筛选正负样本
                for pt_idx in range(obj_pts_count):
                    # 跳过无label的点
                    anchor_label = point_label[pt_idx]
                    if anchor_label == -1:
                        continue

                    # 排除自身
                    pf_sim[pt_idx, pt_idx] = -1.0

                    # 4.1 筛选正样本：同label + partfieldfeats高相似度
                    # 先找所有同label的点
                    same_label_mask = (point_label == anchor_label)
                    same_label_mask[pt_idx] = False  # 排除自身
                    same_label_indices = torch.where(same_label_mask)[0]
                    if len(same_label_indices) < self.topk:
                        continue  # 同label点不足

                    # 在同label点中取partfieldfeats相似度最高的topk
                    same_label_sim = pf_sim[pt_idx, same_label_indices]
                    topk_sim, topk_indices = torch.topk(same_label_sim, self.topk)
                    # 过滤相似度极低的点（确保正样本质量）
                    valid_mask = topk_sim > 0.5  # 可调整阈值
                    if valid_mask.sum() < 3:
                        continue
                    positive_indices = same_label_indices[topk_indices[valid_mask]]

                    # 4.2 筛选负样本：不同label + partfieldfeats低相似度
                    # 先找所有不同label的点
                    diff_label_mask = (point_label != anchor_label) & (point_label != -1)
                    diff_label_indices = torch.where(diff_label_mask)[0]
                    if len(diff_label_indices) < self.topk:
                        continue  # 不同label点不足

                    # 在不同label点中取partfieldfeats相似度最低的topk
                    diff_label_sim = pf_sim[pt_idx, diff_label_indices]
                    bottomk_sim, bottomk_indices = torch.topk(diff_label_sim, self.topk, largest=False)
                    # 过滤相似度较高的点（确保负样本质量）
                    valid_mask = bottomk_sim < 0.2  # 可调整阈值
                    if valid_mask.sum() < 3:
                        continue
                    negative_indices = diff_label_indices[bottomk_indices[valid_mask]]

                    # 4.3 计算约束损失
                    pos_sim = net_sim[pt_idx, positive_indices].mean()
                    neg_sim = net_sim[pt_idx, negative_indices].mean()
                    part_loss += F.relu(self.margin - (pos_sim - neg_sim))

            # 归一化损失
            if part_loss > 0:
                total_pts = pt_offset[-1].item()
                part_loss = part_loss / total_pts

        # 总损失
        total_loss = (texts_loss_nonzero_avg + pts_loss_nonzero_avg) / 2.0 + self.part_weight * part_loss
        return total_loss'''

'''
#################### hard 采样约束
import torch
import torch.nn as nn
import torch.nn.functional as F

class HardDistillLossContrastive(nn.Module):
    def __init__(self, hard_ratio=0.3, margin=0.5):
        super(HardDistillLossContrastive, self).__init__()
        self.hard_ratio = hard_ratio  # 选择硬样本的比例
        self.margin = margin          # 对比损失的边际值

    def forward(self, net_out, pt_offset, mask_embs, mask_pts, logit_scale, is_backward_dist=False, stud_kowledge=None): 
        """
        net_out: [n_total_pts, dim_ft]
        pt_offset: [BS+1,] # 每个物体点索引的起始位置
        mask_embs: [n_totak_masks, dim_ft] 掩码的嵌入向量
        mask_pts: list of [n_cur_masks, n_pts_cur_obj] 二进制掩码
        """
        N_MASKS_TOTAL, N_DIM = mask_embs.shape
        mask_npts = [torch.sum(mask_pt, dim=1).view(-1,1) for mask_pt in mask_pts]
        all_masks_npts = torch.cat(mask_npts).cuda()
        mask_nopts = ((all_masks_npts==0)*1).squeeze()
        
        # 计算每个掩码的平均特征
        obj_start_idxs = torch.cat((torch.tensor([0]).cuda(), pt_offset[:-1]))
        sum_feats = [mask_pt.cuda()*1.0 @ net_out[start_idx:end_idx,:] 
                    for (mask_pt, start_idx, end_idx) in zip(mask_pts, obj_start_idxs, pt_offset)]
        all_sum_feats = torch.cat(sum_feats)
        all_mask_avg_feats = all_sum_feats / (all_masks_npts+1e-12)  # n_total_masks, out_dim

        # 计算相似度分数
        logits = mask_embs @ all_mask_avg_feats.T * torch.exp(logit_scale)
        target = torch.arange(N_MASKS_TOTAL).cuda()
        modified_target = mask_nopts * -100 + (1-mask_nopts)*target

        # 计算原始交叉熵损失
        texts_loss = F.cross_entropy(logits, modified_target, reduction='none')
        pts_loss = F.cross_entropy(logits.T, modified_target, reduction='none')

        # 硬样本挖掘 - 文本损失部分
        if texts_loss.sum() > 0:
            # 过滤掉零值损失
            valid_texts_loss = texts_loss[texts_loss > 0]
            # 按损失值排序，选取最大的hard_ratio比例作为硬样本
            num_hard = max(1, int(len(valid_texts_loss) * self.hard_ratio))
            hard_texts_loss, _ = torch.topk(valid_texts_loss, num_hard)
            texts_loss_nonzero_avg = hard_texts_loss.mean()
        else:
            texts_loss_nonzero_avg = torch.tensor(0.0, device=logits.device)

        # 硬样本挖掘 - 点损失部分
        if pts_loss.sum() > 0:
            # 过滤掉零值损失
            valid_pts_loss = pts_loss[pts_loss > 0]
            # 按损失值排序，选取最大的hard_ratio比例作为硬样本
            num_hard = max(1, int(len(valid_pts_loss) * self.hard_ratio))
            hard_pts_loss, _ = torch.topk(valid_pts_loss, num_hard)
            pts_loss_nonzero_avg = hard_pts_loss.mean()
        else:
            pts_loss_nonzero_avg = torch.tensor(0.0, device=logits.device)

        # 计算对比损失的硬样本惩罚项
        # 获取正负样本对的掩码
        positive_mask = torch.eye(N_MASKS_TOTAL, device=logits.device)
        negative_mask = 1 - positive_mask
        
        # 计算相似度
        sim = F.softmax(logits, dim=1)
        
        # 正样本对的相似度应尽可能高，负样本对的相似度应尽可能低
        positive_sim = (positive_mask * sim).sum(dim=1)
        negative_sim = (negative_mask * sim).sum(dim=1)
        
        # 硬负样本：与锚点过于相似的负样本
        hard_neg_mask = negative_sim > (positive_sim - self.margin)
        # 硬正样本：与锚点不够相似的正样本
        hard_pos_mask = positive_sim < (negative_sim - self.margin)
        
        # 对硬样本添加额外惩罚
        hard_penalty = 0.0
        if hard_neg_mask.sum() > 0 or hard_pos_mask.sum() > 0:
            hard_penalty = (F.relu(negative_sim[hard_neg_mask] - (positive_sim[hard_neg_mask] - self.margin)).mean() +
                           F.relu((negative_sim[hard_pos_mask] + self.margin) - positive_sim[hard_pos_mask]).mean()) / 2
        
        # 总损失 = 硬样本平均损失 + 硬样本惩罚项
        loss = (texts_loss_nonzero_avg + pts_loss_nonzero_avg) / 2.0 + hard_penalty

        return loss'''
