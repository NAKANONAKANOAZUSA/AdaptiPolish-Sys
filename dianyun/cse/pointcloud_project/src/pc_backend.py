"""
pc_backend.py
作用：
  1) 在没有 open3d 的环境中读取点云（支持 PLY）
  2) 在没有可视化库的环境中，保存预测/真值为带颜色的 PLY
说明：
  - 这是“临时纯 Python 后端”
  - 在 RISC 上能直接跑
  - 第 2 步我们会把这里替换成 C++ 点云库后端（pybind11）
"""

import numpy as np


def load_pointcloud(path: str) -> np.ndarray:
    """
    读取 ASCII PLY 点云文件，返回 xyz (N,3) float32
    目前支持最常见的 ASCII 格式：
      - header 里有 element vertex N
      - 之后每行至少 3 列 (x y z)
    """
    if not path.lower().endswith(".ply"):
        raise ValueError("目前只支持 .ply ASCII 文件")

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    # 1) 找 vertex 数量
    num_verts = None
    header_end = None
    for i, line in enumerate(lines):
        line = line.strip()
        if line.startswith("element vertex"):
            num_verts = int(line.split()[-1])
        if line == "end_header":
            header_end = i + 1
            break

    if num_verts is None or header_end is None:
        raise ValueError("PLY header 解析失败：没找到 vertex 或 end_header")

    # 2) 读取后面的点
    pts = []
    for line in lines[header_end:header_end + num_verts]:
        parts = line.strip().split()
        if len(parts) < 3:
            continue
        x, y, z = map(float, parts[:3])
        pts.append([x, y, z])

    xyz = np.asarray(pts, dtype=np.float32)
    return xyz


def save_colored_ply(path: str, xyz: np.ndarray, labels: np.ndarray):
    """
    保存带颜色的 ASCII ply。
    labels 数值：
      0 -> 灰色 (环境)
      1 -> 蓝色 (工件)
      2 -> 红色 (瑕疵)
    """
    assert xyz.ndim == 2 and xyz.shape[1] == 3
    assert labels.ndim == 1 and labels.shape[0] == xyz.shape[0]

    # 颜色映射
    colors = np.zeros((labels.shape[0], 3), dtype=np.uint8)
    colors[labels == 0] = [128, 128, 128]
    colors[labels == 1] = [0, 0, 255]
    colors[labels == 2] = [255, 0, 0]

    N = xyz.shape[0]

    header = [
        "ply",
        "format ascii 1.0",
        f"element vertex {N}",
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        "end_header"
    ]

    with open(path, "w", encoding="utf-8") as f:
        for h in header:
            f.write(h + "\n")
        for (x, y, z), (r, g, b) in zip(xyz, colors):
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b}\n")

    print(f"✅ 已保存带颜色的 ply: {path}")
