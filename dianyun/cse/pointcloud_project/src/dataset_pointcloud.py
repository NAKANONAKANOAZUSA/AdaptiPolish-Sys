import numpy as np
import os
import torch
from torch.utils.data import Dataset

# 新增：导入 C++ 后端
from dianyun.cse.pointcloud_project.src.pc_backend import load_pointcloud   # ← 这是 C++ 的接口（）

class PointCloudDataset(Dataset):
    def __init__(self, root, num_points=4096):
        self.root = root
        self.num_points = num_points
        self.files = [f for f in os.listdir(root) if f.endswith(".npz")]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = os.path.join(self.root, self.files[idx])
        data = np.load(path)
        xyz = data["points"].astype(np.float32)
        labels = data["labels"].astype(np.int64)

        # 随机采样
        if xyz.shape[0] > self.num_points:
            idx = np.random.choice(xyz.shape[0], self.num_points, replace=False)
            xyz = xyz[idx]
            labels = labels[idx]

        return torch.from_numpy(xyz), torch.from_numpy(labels)

    # --------------------------
    # 新增：从 PLY 读取点云（for inference）
    # --------------------------
    @staticmethod
    def load_from_ply(path: str):
        xyz = load_pointcloud(path)  # 调用 C++ 加载（）
        return torch.from_numpy(xyz.astype(np.float32))
