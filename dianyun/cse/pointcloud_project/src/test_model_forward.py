"""
test_model_forward.py
作用：
  1. 从 DataLoader 取一个 batch 点云
  2. 喂进 SimplePointNetSeg 模型
  3. 打印输入/输出的形状，确认网络可以在 GPU 上正常前向传播
"""

import torch
from torch.utils.data import DataLoader

from dataset_pointcloud import PointCloudDataset
from model_pointnet import SimplePointNetSeg


def main():
    # 1. 设备设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("当前设备:", device)

    # 2. 数据集 & DataLoader（和之前保持一致）
    data_root = r"/home/er/Desktop/cse/pointcloud_project/data/train"
    dataset = PointCloudDataset(data_root=data_root, num_points=4096)

    dataloader = DataLoader(
        dataset,
        batch_size=2,      # 你现在只有一个 npz 文件，所以其实只会拿到 1 个样本
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    # 3. 创建模型，并放到 GPU 上
    num_classes = 3  # 类别数：0=环境, 1=工件, 2=瑕疵
    model = SimplePointNetSeg(num_classes=num_classes).to(device)
    model.eval()  # 测试用，关掉 dropout/bn 的训练行为

    # 4. 取一个 batch 测试
    for batch_idx, (xyz, label) in enumerate(dataloader):
        print(f"\nbatch_idx = {batch_idx}")
        print("  原始 xyz 形状:", xyz.shape)    # (B, N, 3)
        print("  原始 label 形状:", label.shape)  # (B, N)

        # 把数据移动到 GPU
        xyz = xyz.to(device)      # (B, N, 3)
        label = label.to(device)  # (B, N)

        # 模型期望输入是 (B, 3, N)，所以需要转置一下
        xyz_transposed = xyz.transpose(1, 2)  # (B, 3, N)
        print("  转置后 xyz 形状:", xyz_transposed.shape)

        # 前向传播
        with torch.no_grad():
            pred = model(xyz_transposed)     # (B, N, num_classes)

        print("  模型输出 pred 形状:", pred.shape)

        # 简单看一下每个点的预测类别（取最大值所在的索引）
        pred_labels = pred.argmax(dim=-1)    # (B, N)
        print("  预测标签形状:", pred_labels.shape)

        # 只测试一个 batch 就够了
        break

    print("\n✅ 模型前向传播测试完成，如果上面形状都正常，就可以进入下一步：写训练脚本。")


if __name__ == "__main__":
    main()

