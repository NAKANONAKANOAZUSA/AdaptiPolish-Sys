# api_pointcloud.py

from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np
import torch

from model_pointnet import SimplePointNetSeg
from pc_backend import load_pointcloud, save_colored_ply


class PointCloudSegAPI:
    """
    点云分割 API 封装：
      - 加载 PointNet 分割模型
      - 对输入点云做中心化 + 单位球归一化 + 采样/补齐
      - 执行前向推理，输出每个点的类别
      - 可选：保存带颜色的 PLY 结果

    类别约定（和训练保持一致）：
      0 = 环境
      1 = 工件
      2 = 瑕疵
    """

    def __init__(
        self,
        model_path: str,
        num_classes: int = 3,
        num_points: int = 4096,
        device: Optional[str] = None,
    ) -> None:
        """
        :param model_path: 训练好的权重路径，如 'checkpoints/pointnet_seg_best.pth'
        :param num_classes: 类别数（默认 3）
        :param num_points: 推理时采样/补齐的点数（需要和训练时一致）
        :param device: 'cuda' / 'cpu' / None（自动）
        """
        self.model_path = model_path
        self.num_classes = num_classes
        self.num_points = num_points

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = self._load_model()

    # ---------- 内部工具函数 ----------

    def _load_model(self) -> torch.nn.Module:
        if not os.path.isfile(self.model_path):
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")

        model = SimplePointNetSeg(num_classes=self.num_classes).to(self.device)
        state = torch.load(self.model_path, map_location=self.device)
        model.load_state_dict(state)
        model.eval()
        return model

    @staticmethod
    def _preprocess_points(
        points: np.ndarray,
        num_points: int,
    ) -> np.ndarray:
        """
        预处理点云：
          1) 中心化
          2) 单位球归一化
          3) 采样/补齐到 num_points

        :param points: (N, 3) numpy 数组
        :param num_points: 目标点数
        :return: (num_points, 3) numpy 数组
        """
        xyz = np.asarray(points, dtype=np.float32)

        if xyz.ndim != 2 or xyz.shape[1] != 3:
            raise ValueError(f"预期点云形状为 (N,3)，但收到 {xyz.shape}")

        # 1) 中心化
        centroid = xyz.mean(axis=0, keepdims=True)
        xyz = xyz - centroid

        # 2) 单位球归一化
        m = np.max(np.sqrt((xyz ** 2).sum(axis=1)))
        if m > 0:
            xyz = xyz / m

        # 3) 随机采样/补齐
        N = xyz.shape[0]
        if N >= num_points:
            idx = np.random.choice(N, num_points, replace=False)
        else:
            idx = np.random.choice(N, num_points, replace=True)

        return xyz[idx]

    def _forward(
        self,
        xyz_processed: np.ndarray,
    ) -> np.ndarray:
        """
        执行一次前向推理。

        :param xyz_processed: (num_points,3) numpy 数组
        :return: (num_points,) 预测类别
        """
        # 转 Tensor，形状 (1,3,num_points)
        xyz_tensor = torch.from_numpy(xyz_processed).unsqueeze(0).to(self.device)  # (1, N, 3)
        xyz_tensor = xyz_tensor.transpose(1, 2)  # (1, 3, N)

        with torch.no_grad():
            pred = self.model(xyz_tensor)  # (1, N, num_classes)

        pred_labels = pred.argmax(dim=-1).squeeze(0).cpu().numpy()  # (N,)
        return pred_labels

    # ---------- 对外 API ----------

    def predict_points(
        self,
        points: np.ndarray,
        *,
        save_colored_to: Optional[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        对内存中的点云（N,3）做推理。

        :param points: 原始点云 (N,3)
        :param save_colored_to: 若不为 None，则把 (num_points,3) + 预测类别 写成带颜色 PLY
        :return: (pred_labels, xyz_processed)
                 pred_labels: (num_points,) 预测标签
                 xyz_processed: (num_points,3) 预处理后点云（已采样/归一化）
        """
        xyz_processed = self._preprocess_points(points, self.num_points)
        pred_labels = self._forward(xyz_processed)

        if save_colored_to is not None:
            # 利用你已有的 pc_backend.save_colored_ply
            save_colored_ply(save_colored_to, xyz_processed, pred_labels)

        return pred_labels, xyz_processed

    def predict_ply(
        self,
        ply_path: str,
        *,
        out_ply: Optional[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        从 PLY 文件读取点云并推理。

        :param ply_path: 输入 PLY 文件路径
        :param out_ply: 若不为 None，则保存带颜色的 PLY 到该路径
        :return: (pred_labels, xyz_processed)
        """
        if not os.path.isfile(ply_path):
            raise FileNotFoundError(f"PLY 文件不存在: {ply_path}")

        # 用你的后端读取点云
        xyz_raw = load_pointcloud(ply_path)

        if xyz_raw is None or len(xyz_raw) == 0:
            raise RuntimeError(f"从 {ply_path} 读取到的点云为空")

        xyz_processed = self._preprocess_points(xyz_raw, self.num_points)
        pred_labels = self._forward(xyz_processed)

        if out_ply is not None:
            save_colored_ply(out_ply, xyz_processed, pred_labels)

        return pred_labels, xyz_processed
