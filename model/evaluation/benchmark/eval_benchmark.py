import os

import torch
import argparse
from torch.utils.data import DataLoader
from model.data.data import EvalData3D, EvalShapeNetPart, EvalPartNetE, collate_fn
import numpy as np
from model.evaluation.core import compute_3d_iou_upsample, compute_3d_iou_upsample_pro
import time
from model.evaluation.utils import set_seed, load_model


def evaluate3d(model, dataloader, panoptic=False, N_CHUNKS=1, visualize_seg=False, visualize_all_heatmap=False, savedir=None): # evaluate loader can only have batch size=1
    temperature = np.exp(model.ln_logit_scale.item())
    iou_full_list = []
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            for key in data.keys():
                if isinstance(data[key], torch.Tensor) and "full" not in key:
                    data[key] = data[key].cuda(non_blocking=True)

            net_out = model(x=data)
            text_embeds = data['label_embeds']
            gt_full = data["gt_full"]
            xyz_sub = data["coord"]
            xyz_full = data["xyz_full"]
            cat = data["class_name"][0]
            


            '''########### debug

            print('text_embeds:', text_embeds.shape)
            from release_tmp.augcheck.augcheck import SimilarityAnalyzer
            analyzer = SimilarityAnalyzer('results/find3d_d3compat_prosegloss/ckpt_200.pth')
            save_path = "/apdcephfs_cq11/share_303570626/lanejin/project/Find3D/results_tmp/augcheck/augpartnet.ply"
            best_group, best_sim = analyzer.process_object_debug(
                data, model, text_embeds, save_path=save_path,
                use_advanced_clustering=True, n_clusters=8
            )
            asdf'''




            # if savedir is not None:
            #     os.makedirs(os.path.join(savedir, cat), exist_ok=True)
            #     savepath = os.path.join(savedir, cat, str(i)+".ply")
            # else:
            #     savepath = None
            # try:
            #     # Limit number of saved segmentation results: if os.path.join(savedir, cat) has 10+ files, set savepath = None
            #     if os.path.exists(os.path.join(savedir, cat)) and len(os.listdir(os.path.join(savedir, cat))) >= 10:
            #         savepath = None
            # except:
            #     print('Failed to limit saved segmentation results')
            # uidname = data["uidname"][0]
            # savepath = f'/apdcephfs_cq11/share_303570626/lanejin/project/Find3D/results_tmp/infer_objaverse/{uidname}.ply'
            savepath = None
            full_miou, _ = compute_3d_iou_upsample(net_out, # n_subsampled_pts, feat_dim
                                               text_embeds, # n_parts, feat_dim
                                               temperature,
                                               cat,
                                               xyz_sub,
                                               xyz_full, # n_pts, 3
                                               gt_full, # n_pts,
                                               panoptic=panoptic,
                                               N_CHUNKS=N_CHUNKS,
                                               visualize_seg=visualize_seg,
                                               savepath = savepath,
                                               visualize_all_heatmap=visualize_all_heatmap,
                                               xyz_visualization = data["xyz_visualization"])
            '''full_miou, _ = compute_3d_iou_upsample_pro(net_out, # n_subsampled_pts, feat_dim
                                               text_embeds, # n_parts, feat_dim
                                               temperature,
                                               cat,
                                               xyz_sub,
                                               xyz_full, # n_pts, 3
                                               gt_full, # n_pts,
                                               panoptic=panoptic,
                                               N_CHUNKS=N_CHUNKS,
                                               visualize_seg=visualize_seg,
                                               savepath = savepath,
                                               xyz_visualization = data["xyz_visualization"])'''
            
            print(cat, full_miou)
            iou_full_list += [full_miou]
    full_miou = np.mean(iou_full_list)
    return full_miou

def eval_category_partnete(data_root, category, model, apply_rotation=False, subset=False, decorated=True, visualize_seg=False, visualize_all_heatmap=False, c=None, savedir=None):
    test_data = EvalPartNetE(data_root, category, apply_rotation=apply_rotation, subset=subset, decorated=decorated)
    test_loader = DataLoader(test_data, 
                             batch_size=1, 
                             shuffle=False,
                             collate_fn=collate_fn, 
                             num_workers=0, 
                             drop_last=False)
    stime = time.time()
    full_miou = evaluate3d(model, test_loader, panoptic=False, N_CHUNKS=20, visualize_seg=visualize_seg, visualize_all_heatmap=visualize_all_heatmap, savedir =savedir )
    print(f"{category}: miou: {full_miou}")
    etime = time.time()
    print('time:', etime-stime, ' s')
    return category, full_miou, etime-stime


def eval_category_shapenetpart(data_root, category, model, apply_rotation=False, subset=False, decorated=True, use_tuned_prompt=False, visualize_seg=False, visualize_all_heatmap=False):
    test_data = EvalShapeNetPart(data_root, category, apply_rotation=apply_rotation, subset=subset, decorated=decorated, use_tuned_prompt=use_tuned_prompt)
    test_loader = DataLoader(test_data, 
                             batch_size=1, 
                             shuffle=False,
                             collate_fn=collate_fn, 
                             num_workers=0, 
                             drop_last=False)
    stime = time.time()
    full_miou = evaluate3d(model, test_loader, panoptic=True, N_CHUNKS=1, visualize_seg=visualize_seg, visualize_all_heatmap=visualize_all_heatmap)
    print(f"{category}: miou: {full_miou}")
    etime = time.time()
    print(etime-stime)
    return full_miou, etime-stime


def eval_objaverse(split, data_root, model, decorated=True, use_tuned_prompt=False, visualize_seg=False, visualize_all_heatmap=False, save_dir=None):
    test_data = EvalData3D(split=split, root=data_root, decorated=decorated, use_tuned_prompt=use_tuned_prompt, visualization=visualize_seg or visualize_all_heatmap)
    test_loader = DataLoader(test_data, 
                             batch_size=1, 
                             shuffle=False,
                             collate_fn=collate_fn, 
                             num_workers=0, 
                             drop_last=False)
    stime = time.time()
    full_miou = evaluate3d(model, test_loader, panoptic=False, N_CHUNKS=1, visualize_seg=visualize_seg, visualize_all_heatmap=visualize_all_heatmap)
    print(f"{split}: miou: {full_miou}")
    etime = time.time()
    print(etime-stime)

def eval_partnete(data_root, model, apply_rotation, subset, decorated, cfv=False, save_dir=None):
    partnete_categories = {'Bottle':16, 'Box':17, 'Bucket':18, 'Camera':19, 'Cart':20, 'Chair':21, 'Clock':22,
            "CoffeeMachine": 23, 'Dishwasher': 24, 'Dispenser': 25, "Display": 26, 'Eyeglasses': 27,
            'Faucet': 28, "FoldingChair": 29, "Globe": 30, "Kettle":31, "Keyboard": 32, "KitchenPot": 33,
            "Knife": 34, "Lamp": 35, "Laptop": 36, "Lighter": 37, "Microwave": 38, "Mouse": 39, "Oven": 40,
            "Pen": 41, "Phone": 42, "Pliers": 43, "Printer": 44, "Refrigerator": 45, "Remote": 46,
            "Safe": 47, "Scissors": 48, "Stapler": 49, "StorageFurniture": 50, "Suitcase": 51,
            "Switch": 52, "Table": 53, "Toaster": 54, "Toilet": 55, "TrashCan": 56, "USB": 57,
            "WashingMachine": 58, "Window": 59, "Door": 60}
    full_mious = []
    time_all = 0
    '''for cat in partnete_categories:
        category, full_miou, time_cur = eval_category_partnete(data_root, cat, model, apply_rotation=apply_rotation, subset=subset, decorated=decorated, visualize_seg=visualize_seg)
        full_mious.append(full_miou)
        time_all += time_cur
    full_miou_avg = np.mean(full_mious)
    print(f"miou {full_miou_avg}")
    print(f"time {time_all}")'''
    # Store results per category (category name + iou value)
    results = []
    full_mious = []
    time_all = 0.0

    for cat in partnete_categories:
        # 执行评估，获取类别名称、iou值、耗时
        category, full_miou, time_cur = eval_category_partnete(
            data_root, cat, model, 
            apply_rotation=apply_rotation, 
            subset=subset, 
            decorated=decorated, 
            savedir = save_dir
        ) # visualize_seg=visualize_seg,
        # Round iou to 4 decimal places
        full_miou_rounded = round(full_miou, 4)
        # Store result
        results.append((category, full_miou_rounded))
        full_mious.append(full_miou)  # Keep raw value for computing average
        time_all += time_cur

    # Compute average iou (rounded to 4 decimal places)
    full_miou_avg = round(np.mean(full_mious), 4)
    # Print table
    # Table header
    print(f"{'Category':<15} {'iou':<10}")
    print("-" * 25)
    # Print result per category row by row
    for category, miou in results:
        print(f"{category:<15} {miou:<10.4f}")
    # Separator line
    print("-" * 25)
    # Print average in last row
    print(f"{'Average':<15} {full_miou_avg:<10.4f}")

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "result_table.txt"), "w") as f:
            f.write(f"{'Category':<15} {'iou':<10}\n")
            f.write("-" * 25 + "\n")
            for category, miou in results:
                f.write(f"{category:<15} {miou:<10.4f}\n")
            f.write("-" * 25 + "\n")
            f.write(f"{'Average':<15} {full_miou_avg:<10.4f}\n")



def eval_shapenetpart(data_root, model, apply_rotation, subset, decorated, use_tuned_prompt, visualize_seg=False):
    shapenetpart_categories = {'airplane': 0, 'bag': 1, 'cap': 2, 'car': 3, 'chair': 4, 
        'earphone': 5, 'guitar': 6, 'knife': 7, 'lamp': 8, 'laptop': 9, 
        'motorbike': 10, 'mug': 11, 'pistol': 12, 'rocket': 13, 'skateboard': 14, 'table': 15}
    full_mious = []
    time_all = 0
    for cat in shapenetpart_categories:
        full_miou, time_cur = eval_category_shapenetpart(data_root, cat, model, apply_rotation=apply_rotation, subset=subset, decorated=decorated, visualize_seg=visualize_seg, use_tuned_prompt = use_tuned_prompt)
        full_mious.append(full_miou)
        time_all += time_cur
    full_miou_avg = np.mean(full_mious)
    print(f"full miou {full_miou_avg}")
    print(f"time {time_all}")

       
if __name__ == '__main__':
    set_seed(123)
    parser = argparse.ArgumentParser(description="Please specify a benchmark name and evaluation configurations")
    parser.add_argument("--benchmark", required=True, type=str, help='The benchmark to evaluate on. Should be Objaverse, ShapeNetPart, or PartNetE')
    parser.add_argument("--data_root", required=True, type=str, help='Root directory of the benchmark data')
    parser.add_argument("--save_dir", required=True, type=str, help='results save to')
    parser.add_argument("--checkpoint_path", required=True, type=str, help='path of the checkpoint to evaluate')
    parser.add_argument("--objaverse_split", type=str, help='If benchmark is Objaverse, specify "seenclass", "unseen" or "shapenetpart')
    parser.add_argument("--canonical", action='store_false', dest="rotate", help="whether to perform random rotation - this only applies to ShapeNetPart or PartNetE which have canonical orientations")
    parser.add_argument("--subset", action='store_true', dest="subsample", help="whether to evaluate on subset - this only applies to ShapeNetPart of PartNetE")
    parser.add_argument("--part_query", action='store_false', dest="decorate", help="if true, evaluate with {part} of a {object} as query prompt; if false, evaluate with {part} as query prompt")
    parser.add_argument("--use_shapenetpart_topk_prompt", action='store_true', help="This only applies to ShapeNetPart or Objaverse-ShapeNetPart. Whether to use the topk prompt following PointCLIPV2's procedures to choose prompts")
    parser.set_defaults(rotate=False, subsample=False, decorate=True, use_shapenetpart_topk_prompt=False)  # Default: no rotation augmentation
    args = parser.parse_args()
    
    model = load_model(args.checkpoint_path)
    
    if args.benchmark == "Objaverse":
        if not args.objaverse_split:
            print("If evaluating on Objaverse, please specify split- seenclass/unseen/shapenetpart")
        elif args.objaverse_split not in ["seenclass", "unseen", "shapenetpart"]:
            print("If evaluating on Objaverse, please choose a split from seenclass, unseen, shapenetpart")
        else:
            eval_objaverse(args.objaverse_split, args.data_root, model, decorated=args.decorate, use_tuned_prompt=args.use_shapenetpart_topk_prompt, save_dir=args.save_dir)
    elif args.benchmark == "ShapeNetPart":
        eval_shapenetpart(args.data_root, model, apply_rotation=args.rotate, subset=args.subsample, decorated=args.decorate, use_tuned_prompt = args.use_shapenetpart_topk_prompt)
    elif args.benchmark == "PartNetE":
        eval_partnete(args.data_root, model, apply_rotation=args.rotate, subset=args.subsample, decorated=args.decorate, save_dir=args.save_dir)
    else:
        print("Invalid benchmark. Please choose one of Objaverse, ShapeNetPart, or PartNetE.")
    
    