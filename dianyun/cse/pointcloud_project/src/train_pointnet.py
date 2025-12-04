"""
train_pointnet.py (improved)

改进点：
1) 类感知采样：采样4096点时，优先保留瑕疵点（类2），再随机补齐。
2) 类权重改为 sqrt(1/count) + clamp，避免权重极端不稳定。
"""

import os
import time
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dianyun.cse.pointcloud_project.src.dataset_pointcloud import PointCloudDataset
from dianyun.cse.pointcloud_project.src.model_pointnet import SimplePointNetSeg


def compute_class_weights(dataset, num_classes=3, max_ratio=10.0):
    """
    改进版类别权重：
      w = 1/sqrt(count)
      再归一化到均值=1
      再 clamp 最大权重倍数，避免极端不稳定
    """
    counts = np.zeros(num_classes, dtype=np.float64)

    print("\n🔢 正在统计整个数据集中各类别的点数...")
    for idx in range(len(dataset)):
        _, label = dataset[idx]
        label_np = label.numpy()
        for c in range(num_classes):
            counts[c] += (label_np == c).sum()

    print("📊 各类别点数统计：")
    for c in range(num_classes):
        print(f"  类 {c}: {int(counts[c])} 个点")

    # sqrt inverse frequency
    class_weights = 1.0 / np.sqrt(counts + 1e-6)
    class_weights = class_weights / class_weights.mean()

    # clamp to avoid huge ratios
    class_weights = np.clip(class_weights, 1.0 / max_ratio, max_ratio)

    print("\n⚖ 改进后的类别权重（sqrt逆频率 + clamp）：")
    for c in range(num_classes):
        print(f"  类 {c}: {class_weights[c]:.4f}")

    return torch.tensor(class_weights, dtype=torch.float32)


def class_aware_sample(xyz, label, num_points=4096, defect_class=2):
    """
    类感知采样：
    - 优先把 defect_class 的点全选进来（如果超过 num_points 就随机截断）
    - 剩下的点再从非 defect 中随机补齐
    """
    xyz_np = xyz.cpu().numpy()
    label_np = label.cpu().numpy()

    defect_idx = np.where(label_np == defect_class)[0]
    other_idx  = np.where(label_np != defect_class)[0]

    if len(defect_idx) >= num_points:
        chosen_defect = np.random.choice(defect_idx, num_points, replace=False)
        final_idx = chosen_defect
    else:
        # 先放全部瑕疵点
        need = num_points - len(defect_idx)
        chosen_other = np.random.choice(other_idx, need, replace=(len(other_idx) < need))
        final_idx = np.concatenate([defect_idx, chosen_other])

    np.random.shuffle(final_idx)

    xyz_s = torch.from_numpy(xyz_np[final_idx]).to(xyz.device)
    label_s = torch.from_numpy(label_np[final_idx]).to(label.device)
    return xyz_s, label_s


def train():
    # -------- 配置 --------
    data_root = r"C:\Users\SRIT\Desktop\ai\5\pointcloud_project\data\train"
    checkpoint_dir = r"C:\Users\SRIT\Desktop\ai\5\pointcloud_project\checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    num_points = 4096
    num_classes = 3
    batch_size = 2
    num_epochs = 30
    learning_rate = 1e-3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("当前设备:", device)

    # -------- Dataset & Loader --------
    dataset = PointCloudDataset(data_root=data_root, num_points=num_points)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    # -------- 类权重 --------
    class_weights = compute_class_weights(dataset, num_classes=num_classes).to(device)

    # -------- 模型/优化器/loss --------
    model = SimplePointNetSeg(num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    print("\n✅ 开始训练（带瑕疵增强采样）...\n")

    best_loss = float("inf")
    best_model_path = os.path.join(checkpoint_dir, "pointnet_seg_best.pth")

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        total_points = 0
        correct_points = 0

        start_time = time.time()

        for batch_idx, (xyz, label) in enumerate(dataloader):
            xyz = xyz.to(device)     # (B,N,3)
            label = label.to(device) # (B,N)

            # --- 类感知采样：对 batch 内每个样本单独增强 ---
            xyz_list = []
            label_list = []
            for b in range(xyz.shape[0]):
                xyz_s, label_s = class_aware_sample(xyz[b], label[b], num_points=num_points, defect_class=2)
                xyz_list.append(xyz_s)
                label_list.append(label_s)

            xyz = torch.stack(xyz_list, dim=0)       # (B,4096,3)
            label = torch.stack(label_list, dim=0)   # (B,4096)

            xyz_transposed = xyz.transpose(1, 2)     # (B,3,4096)

            pred = model(xyz_transposed)             # (B,4096,3)

            B, N, C = pred.shape
            pred_2d = pred.reshape(B * N, C)
            label_1d = label.reshape(B * N)

            loss = criterion(pred_2d, label_1d)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            total_points += B * N

            with torch.no_grad():
                pred_labels = pred.argmax(dim=-1)
                correct_points += (pred_labels == label).sum().item()

            if (batch_idx + 1) % 10 == 0:
                print(f"  [Epoch {epoch:03d}] Batch {batch_idx+1:03d} | Loss: {loss.item():.4f}")

        avg_loss = epoch_loss / len(dataloader)
        acc = correct_points / total_points

        elapsed = time.time() - start_time
        print(f"\n📎 Epoch {epoch:03d}/{num_epochs} 完成 | "
              f"平均 Loss: {avg_loss:.4f} | 点级精度: {acc*100:.2f}% | "
              f"用时: {elapsed:.1f} 秒")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"💾 已保存当前最优模型到: {best_model_path}\n")
        else:
            print("（本轮没有超越最优模型）\n")

    print("🎉 训练结束！")
    print(f"最优平均 Loss: {best_loss:.4f}")
    print(f"最优模型已保存在: {best_model_path}")


if __name__ == "__main__":
    train()
