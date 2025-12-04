"""
model_pointnet.py
一个简化版的 PointNet 点云分割模型，用于每个点的分类（0/1/2）。

输入形状：
    x: (B, 3, N)   # B=batch_size, N=点数

输出形状：
    pred: (B, N, num_classes)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimplePointNetSeg(nn.Module):
    def __init__(self, num_classes=3):
        super(SimplePointNetSeg, self).__init__()

        # --------- 特征提取部分（对每个点做 MLP）---------
        # Conv1d 的 kernel_size=1，其实相当于对每个点做一个全连接层
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)

        self.conv2 = nn.Conv1d(64, 128, 1)
        self.bn2 = nn.BatchNorm1d(128)

        self.conv3 = nn.Conv1d(128, 256, 1)
        self.bn3 = nn.BatchNorm1d(256)

        # 再升一点维度，作为提取“全局特征”的基础
        self.conv_global = nn.Conv1d(256, 512, 1)
        self.bn_global = nn.BatchNorm1d(512)

        # --------- 分割头（结合点特征 + 全局特征，再输出每个点的类别）---------
        # 输入通道：局部 256 + 全局 512 = 768
        self.conv4 = nn.Conv1d(256 + 512, 256, 1)
        self.bn4 = nn.BatchNorm1d(256)

        self.conv5 = nn.Conv1d(256, 128, 1)
        self.bn5 = nn.BatchNorm1d(128)

        self.dropout = nn.Dropout(p=0.3)

        # 最后输出 num_classes（这里是 3 类：0/1/2）
        self.conv6 = nn.Conv1d(128, num_classes, 1)

    def forward(self, x):
        """
        x: (B, 3, N) 的输入点云
        return:
            (B, N, num_classes) 的每点分类 logits（还没有过 softmax）
        """
        B, C, N = x.shape  # C 应该是 3

        # ---- 点特征提取 ----
        x = F.relu(self.bn1(self.conv1(x)))   # (B, 64, N)
        x = F.relu(self.bn2(self.conv2(x)))   # (B, 128, N)
        x = F.relu(self.bn3(self.conv3(x)))   # (B, 256, N)
        point_feat = x                        # 保存一下局部特征

        # ---- 全局特征提取 ----
        x = F.relu(self.bn_global(self.conv_global(x)))  # (B, 512, N)

        # 对所有点做 max pooling，得到一个全局特征向量 (B, 512, 1)
        global_feat = torch.max(x, dim=2, keepdim=True)[0]  # (B, 512, 1)

        # 把全局特征复制 N 次，拼到每个点上
        global_feat_expanded = global_feat.repeat(1, 1, N)  # (B, 512, N)

        # 局部(256) + 全局(512) = 768 维特征
        x = torch.cat([point_feat, global_feat_expanded], dim=1)  # (B, 768, N)

        # ---- 分割头（输出每个点的类别）----
        x = F.relu(self.bn4(self.conv4(x)))  # (B, 256, N)
        x = F.relu(self.bn5(self.conv5(x)))  # (B, 128, N)
        x = self.dropout(x)
        x = self.conv6(x)                    # (B, num_classes, N)

        # 调整维度为 (B, N, num_classes)，方便和 label 对齐
        x = x.transpose(1, 2).contiguous()   # (B, N, num_classes)

        return x
