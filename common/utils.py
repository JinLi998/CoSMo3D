import torch
import numpy as np
import torch.nn.functional as F
import torch
import open3d as o3d
import matplotlib.pyplot as plt
import plotly.graph_objects as go


def visualize_pts(points, colors, save_path=None, save_rendered_path=None):

    # 获取点云和颜色数据（假设points和colors是形状为[N, 3]的张量）
    points_np = points.cpu().numpy()  # 形状：(N, 3)，xyz坐标
    colors_np = colors.cpu().numpy()  # 形状：(N, 3)，rgb颜色（需确保值在0-255范围，且为整数）

    # 确保颜色值为0-255的整数（若原颜色是归一化到0-1的浮点数，需转换）
    if colors_np.dtype == np.float32 or colors_np.dtype == np.float64:
        colors_np = (colors_np * 255).astype(np.uint8)
    else:
        colors_np = colors_np.astype(np.uint8)

    # 保存为带颜色的PLY文件
    ply_path = f"{save_path}colored_point_cloud.ply"
    with open(ply_path, 'w') as f:
        # PLY文件头
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points_np)}\n")  # 点的数量
        # 定义属性（x, y, z为浮点数；red, green, blue为0-255整数）
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        
        # 写入点数据（每行：x y z r g b）
        for i in range(len(points_np)):
            x, y, z = points_np[i]
            r, g, b = colors_np[i]
            f.write(f"{x} {y} {z} {r} {g} {b}\n")

    print(f"带颜色的PLY文件已保存至：{ply_path}")

    
    # if save_path:
    #     np.save(f"{save_path}xyz.npy", points.cpu().numpy())
    #     np.save(f"{save_path}rgb.npy", colors.cpu().numpy())
    # 
    # points = points.cpu().numpy()
    # fig = go.Figure(data=[go.Scatter3d(
    #     x=points[:, 0],
    #     y=points[:, 1],
    #     z=points[:, 2],
    #     mode='markers',
    #     marker=dict(
    #         size=1.7,
    #         color=(colors.cpu().numpy()*255).astype(int),  # Use RGB colors
    #         opacity=0.99
    #     ))])
    # x_min, x_max = -2,2#points[:, 0].min(), points[:, 0].max()
    # y_min, y_max = -2,2#points[:, 1].min(), points[:, 1].max()
    # z_min, z_max = -2,2#points[:, 2].min(), points[:, 2].max()
    # fig.update_layout(
    #     scene=dict(
    #         bgcolor='rgb(220, 220, 220)'  # Set the 3D scene background to light grey
    #     ),
    #     paper_bgcolor='rgb(220, 220, 220)' # Set the overall figure background to light grey
    # )
    # fig.update_layout(
    #     scene=dict(
    #         xaxis=dict(title='x', range=[x_min, x_max], showgrid=False, zeroline=False, visible=False),
    #         yaxis=dict(title='y', range=[y_min, y_max], showgrid=False, zeroline=False, visible=False),
    #         zaxis=dict(title='z', range=[z_min, z_max], showgrid=False, zeroline=False, visible=False),
    #         aspectmode='manual',
    #         aspectratio=dict(
    #             x=(x_max - x_min),
    #             y=(y_max - y_min),
    #             z=(z_max - z_min)
    #         )
    #     ),
    #     scene_camera=dict(
    #         up=dict(x=0, y=1, z=0),  # Adjust these values for your point cloud
    #         eye=dict(x=0, y=-2, z=0),  # Increase the values to move further away
    #         center = dict(x=0,y=0,z=0)
    #     )
    # )
    
    # if save_rendered_path:
    #     fig.write_image(save_rendered_path)
    # else:
    #     fig.show()
    
def visualize_pt_labels(pts, labels, save_path=None, save_rendered_path=None): # pts is n*3, colors is n, 0 - n-1 where 0 is unlabeled
    part_num = labels.max()
    cmap_matrix = torch.tensor([[1,1,1], [1,0,0], [0,1,0], [0,0,1], [1,1,0], [1,0,1],
                [0,1,1], [0.5,0.5,0.5], [0.5,0.5,0], [0.5,0,0.5],[0,0.5,0.5],
                [0.1,0.2,0.3],[0.2,0.5,0.3], [0.6,0.3,0.2], [0.5,0.3,0.5],
                [0.6,0.7,0.2],[0.5,0.8,0.3]])[:part_num+1,:]
    colors = ["white", "red", "green", "blue", "yellow", "magenta", "cyan","grey", "olive",
                "purple", "teal", "navy", "darkgreen", "brown", "pinkpurple", "yellowgreen", "limegreen"]
    caption_list=[f"{i}:{colors[i]}" for i in range(part_num+1)]
    onehot = F.one_hot(labels.long(), num_classes=part_num+1) * 1.0 # n_pts, part_num+1, each row 00.010.0, first place is unlabeled (0 originally)
    pts_rgb = torch.matmul(onehot, cmap_matrix) # n_pts,3
    visualize_pts(pts, pts_rgb, save_path=save_path, save_rendered_path=save_rendered_path)
    return caption_list


def visualize_pt_heatmap(pts, scores, save_path=None): # pts is n*3, scores shape (n,) and is a value between 0 and 1
    pts_rgb = torch.tensor(plt.cm.jet(scores.numpy())[:,:3]).squeeze()
    visualize_pts(pts, pts_rgb, save_path=save_path)


def visualize_pts_subsampled(pts, colors, n_samples):
    perm = torch.randperm(n_samples)
    idx = perm[:n_samples]
    subsampled_pts = pts[idx,:]
    subsampled_colors = colors[idx,:]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(subsampled_pts.numpy())
    pcd.colors = o3d.utility.Vector3dVector(subsampled_colors)
    o3d.visualization.draw_plotly([pcd],
                                  front=[0, 0, -1],
                                  lookat=[0, 0, -1],
                                  up=[0, 1, 0])


def rotate_pts(pts, angles, device=None, return_r=False): # list of points as a tensor, N*3

    roll = angles[0].reshape(1)
    yaw = angles[1].reshape(1)
    pitch = angles[2].reshape(1)

    tensor_0 = torch.zeros(1).to(device)
    tensor_1 = torch.ones(1).to(device)

    RX = torch.stack([
                    torch.stack([tensor_1, tensor_0, tensor_0]),
                    torch.stack([tensor_0, torch.cos(roll), -torch.sin(roll)]),
                    torch.stack([tensor_0, torch.sin(roll), torch.cos(roll)])]).reshape(3,3)

    RY = torch.stack([
                    torch.stack([torch.cos(yaw), tensor_0, torch.sin(yaw)]),
                    torch.stack([tensor_0, tensor_1, tensor_0]),
                    torch.stack([-torch.sin(yaw), tensor_0, torch.cos(yaw)])]).reshape(3,3)

    RZ = torch.stack([
                    torch.stack([torch.cos(pitch), -torch.sin(pitch), tensor_0]),
                    torch.stack([torch.sin(pitch), torch.cos(pitch), tensor_0]),
                    torch.stack([tensor_0, tensor_0, tensor_1])]).reshape(3,3)

    R = torch.mm(RZ, RY)
    R = torch.mm(R, RX)
    if device == "cuda":
        R = R.cuda()
    pts_new = torch.mm(pts, R.T)

    if return_r:
        return pts_new, R
    return pts_new


# 生成旋转矩阵，三个轴都是-1，1
def get_random_xyz_rotation_matrix() -> np.ndarray:
    """
    生成X、Y、Z轴均带随机扰动的3D旋转矩阵，每个轴的旋转角度范围为[-1, 1]度（自动转为弧度）
    无需输入参数，返回最终组合后的3x3旋转矩阵（numpy数组）
    """
    # 1. 定义角度范围：[-1, 1]度，转为弧度（原类逻辑：角度值 * π）
    angle_range = [-1, 1]
    # 2. 为X、Y、Z轴分别生成随机角度（弧度制）
    angle_x = np.random.uniform(angle_range[0], angle_range[1]) * np.pi
    angle_y = np.random.uniform(angle_range[0], angle_range[1]) * np.pi
    angle_z = np.random.uniform(angle_range[0], angle_range[1]) * np.pi

    # 3. 计算各轴旋转矩阵的正弦、余弦值
    # X轴旋转矩阵分量
    cx, sx = np.cos(angle_x), np.sin(angle_x)
    rx = np.array([
        [1, 0, 0],
        [0, cx, -sx],
        [0, sx, cx]
    ])
    # Y轴旋转矩阵分量
    cy, sy = np.cos(angle_y), np.sin(angle_y)
    ry = np.array([
        [cy, 0, sy],
        [0, 1, 0],
        [-sy, 0, cy]
    ])
    # Z轴旋转矩阵分量
    cz, sz = np.cos(angle_z), np.sin(angle_z)
    rz = np.array([
        [cz, -sz, 0],
        [sz, cz, 0],
        [0, 0, 1]
    ])

    # 4. 组合X、Y、Z轴旋转矩阵（旋转顺序：先X→再Y→最后Z，与原类单轴旋转逻辑兼容）
    # 矩阵乘法规则：总旋转矩阵 = Rz * Ry * Rx（右乘表示先应用Rx，再应用Ry，最后应用Rz）
    rot_matrix = np.dot(rz, np.dot(ry, rx))

    return torch.tensor(rot_matrix).float()
