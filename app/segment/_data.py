# augmentation code builds upon https://huggingface.co/Pointcept/PointTransformerV3/blob/main/s3dis-semseg-pt-v3m1-1-ppt-extreme/config.py
from model.data.augmentation import *
import numpy as np


def prep_points_val3d(xyz, rgb, normal, gt, xyz_full, gt_full):
    # the input xyz is expected to be ~5000 points, and the returned coord will be grid-sampled to e.g. 3000
    # the xyz_full can be however dense, e.g. 300k points for partnete, gt_full is the same size as xyz_full
    # but for sparser point clouds we can keep them the same
    # xyz, rgb, normal all (n,3) numpy arrays
    # rgb is 0-255
    # first shift coordinate frame since model is trained on depth coordinate
    xyz_change_axis = np.concatenate([-xyz[:,0].reshape(-1,1), xyz[:,2].reshape(-1,1), xyz[:,1].reshape(-1,1)], axis=1)
    xyz_full_change_axis = np.concatenate([-xyz_full[:,0].reshape(-1,1), xyz_full[:,2].reshape(-1,1), xyz_full[:,1].reshape(-1,1)], axis=1)
    data_dict = {"coord": xyz_change_axis, "color": rgb, "normal":normal, "gt":gt, "xyz_full": xyz_full_change_axis}
    data_dict = CenterShift(apply_z=True)(data_dict)
    data_dict = GridSample(grid_size=0.02,hash_type='fnv',mode='train',return_grid_coord=True)(data_dict) # mode train is used in original code, text will subsample points n times and create many samples out of one sample
    data_dict = CenterShift(apply_z=False)(data_dict)
    data_dict = NormalizeColor()(data_dict)
    data_dict = ToTensor()(data_dict)
    data_dict = Collect(keys=('coord', 'grid_coord', "gt", "xyz_full"),
                        feat_keys=('color', 'normal'))(data_dict)
    data_dict["gt_full"] = gt_full
    return data_dict

