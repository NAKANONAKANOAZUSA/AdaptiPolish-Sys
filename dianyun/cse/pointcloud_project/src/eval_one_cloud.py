"""
eval_one_cloud.py (RISC 兼容版)
作用：
  1. 载入训练好的 PointNet 分割模型
  2. 读取一个测试点云 npz（points + labels）
  3. 做一次前向推理，计算每个类别的准确率
  4. 保存：真值上色 & 预测上色 的 PLY 文件
"""

import os
import numpy as np
import torch

from dianyun.cse.pointcloud_project.src.dataset_pointcloud import PointCloudDataset
from dianyun.cse.pointcloud_project.src.model_pointnet import SimplePointNetSeg
from dianyun.cse.pointcloud_project.src.pc_backend import save_colored_ply


CKPT_PATH = r"/home/er/Desktop/cse/pointcloud_project/checkpoints/pointnet_seg_best.pth"
TEST_DATA_ROOT = r"/home/er/Desktop/cse/hebing/XLJ4/output"
NUM_POINTS = 4096


def main():
    # 1. 设备 & 模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("当前设备:", device)

    if not os.path.isfile(CKPT_PATH):
        raise FileNotFoundError(f"找不到模型文件: {CKPT_PATH}")

    model = SimplePointNetSeg(num_classes=3).to(device)
    state = torch.load(CKPT_PATH, map_location=device)
    model.load_state_dict(state)
    model.eval()
    print("✅ 已加载模型权重。")

    # 2. 构建 Dataset，只取一个样本来测试
    dataset = PointCloudDataset(TEST_DATA_ROOT, num_points=NUM_POINTS)
    print(f"✅ 在 {TEST_DATA_ROOT} 中发现 {len(dataset)} 个 npz 文件。")

    xyz, gt_label = dataset[0]   # xyz:(N,3), gt_label:(N,)

    print("测试样本 xyz 形状:", xyz.shape)
    print("测试样本 gt_label 形状:", gt_label.shape)

    # 3. 前向推理
    xyz_b = xyz.unsqueeze(0).to(device)        # (1, N, 3)
    gt_label = gt_label.to(device)            # (N,)

    with torch.no_grad():
        pred_logits = model(xyz_b.transpose(1, 2))  # (1, N, 3)
        pred_labels = pred_logits.argmax(dim=-1).squeeze(0)  # (N,)

    # 4. 统计整体与每个类别的准确率
    gt_cpu = gt_label.cpu()
    pred_cpu = pred_labels.cpu()

    correct_all = (pred_cpu == gt_cpu).sum().item()
    total_all = gt_cpu.numel()
    acc_all = correct_all / total_all * 100.0

    print(f"\n🔎 整体点级准确率: {acc_all:.2f}% ({correct_all} / {total_all})")

    for cls in [0, 1, 2]:
        mask = (gt_cpu == cls)
        total_cls = mask.sum().item()
        if total_cls == 0:
            print(f"  类 {cls}: 测试点中没有该类。")
            continue
        correct_cls = (pred_cpu[mask] == gt_cpu[mask]).sum().item()
        acc_cls = correct_cls / total_cls * 100.0
        print(f"  类 {cls}: 准确率 {acc_cls:.2f}% ({correct_cls} / {total_cls})")

    # 5. 保存 GT 和 Prediction 的 colored ply
    points_np = xyz.cpu().numpy()   # (N,3)
    gt_np = gt_cpu.numpy()
    pred_np = pred_cpu.numpy()

    save_colored_ply("eval_gt_colored.ply", points_np, gt_np)
    save_colored_ply("eval_pred_colored.ply", points_np, pred_np)

    print("\n✅ 评估完成：已生成 eval_gt_colored.ply 和 eval_pred_colored.ply")


if __name__ == "__main__":
    main()
