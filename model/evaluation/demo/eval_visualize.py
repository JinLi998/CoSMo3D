import os
# 添加解释器路径
import sys
import time
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../..")
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")


import torch
import numpy as np
from model.evaluation.core import visualize_3d_upsample， compute_3d_iou_upsample_pro
import numpy as np
import argparse
from model.evaluation.utils import set_seed, load_model, preprocess_pcd, read_pcd, encode_text
from release_module.obj2ptscolornorm.obj2ptscolornorm import load_and_sample_mesh


# data is a dict that has gone through preprocessing as training (normalizing etc.)
def visualize_seg3d(model, data, mode, N_CHUNKS=5, savepath=None): # evaluate loader can only have batch size=1
    if mode == "segmentation":
        heatmap = False
    elif mode == "heatmap":
        heatmap = True
    else:
        print("unsupported mode")
        return
    start_time = time.time()
    temperature = np.exp(model.ln_logit_scale.item())
    with torch.no_grad():
        for key in data.keys():
            if isinstance(data[key], torch.Tensor) and "full" not in key:
                data[key] = data[key].cuda(non_blocking=True)
        net_out = model(x=data)  # 这个里面只有对点云的处理 ； 无label_embeds
        print(f"-----------------Model inference done in {time.time() - start_time:.2f} seconds")
        text_embeds = data['label_embeds']
        xyz_sub = data["coord"]
        xyz_full = data["xyz_full"]
        visualize_3d_upsample(net_out, # n_subsampled_pts, feat_dim
                            text_embeds, # n_parts, feat_dim
                            temperature,
                            xyz_sub,
                            xyz_full, # n_pts, 3
                            panoptic=False,
                            N_CHUNKS=N_CHUNKS,
                            heatmap=heatmap,
                            savepath=savepath)
        # compute_3d_iou_upsample_pro(net_out, # n_subsampled_pts, feat_dim
        #                     text_embeds, # n_parts, feat_dim
        #                     temperature,
        #                     xyz_sub,
        #                     xyz_full, # n_pts, 3
        #                     panoptic=False,
        #                     N_CHUNKS=N_CHUNKS,
        #                     heatmap=heatmap,
        #                     savepath=savepath)
    return


def eval_obj_wild(model, obj_path, mode, queries, savepath):
    if mode not in ["segmentation", "heatmap"]:
        print("only segmentation or heatmap mode are supported")
        return
    # 如果是pcd文件，读取xyz, rgb, normal
    if obj_path.endswith('.pcd'):
        xyz, rgb, normal = read_pcd(obj_path,visualize=False)  # (torch.Size([5000, 3]), torch.Size([5000, 3]), torch.Size([5000, 3]))
    elif obj_path.endswith('.obj'):
        xyz, rgb, normal = load_and_sample_mesh(obj_path, 5000, colornormflag=True)  # (obj_path, visualize=False, num_points=5000, return_normal=True)  # (torch.Size([5000, 3]), torch.Size([5000, 3]), torch.Size([5000, 3]))

    start_time = time.time()
    data_dict = preprocess_pcd(xyz.cuda(), rgb.cuda(), normal.cuda())
    data_dict["label_embeds"] = encode_text(queries)
    print(f"-----------------Data preprocessed in {time.time() - start_time:.2f} seconds")
    visualize_seg3d(model, data_dict, mode, savepath=savepath)
    return


if __name__ == '__main__':
    set_seed(123)
    parser = argparse.ArgumentParser(description="Please specify input point cloud path and model checkpoint path")
    parser.add_argument("--object_path", required=True, type=str, help='The point cloud to evaluate on. Should be a .pcd file')
    parser.add_argument("--checkpoint_path", required=True, type=str, help='path of the checkpoint to evaluate')
    parser.add_argument("--mode", required=True, type=str, help='segmentation or heatmap')
    parser.add_argument("--queries", required=True, nargs='+', help='list of queries')
    parser.add_argument("--savepath", type=str, default=None, help='保存分割结果的路径')
    args = parser.parse_args()
    

    start_time = time.time()
    model = load_model(args.checkpoint_path)
    print(f"-----------------Model loaded in {time.time() - start_time:.2f} seconds")
    eval_obj_wild(model,args.object_path, args.mode, args.queries, args.savepath)

    