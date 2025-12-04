import os
import torch
import numpy as np  # 新增：用于处理点云坐标与标签
from dianyun.cse.pointcloud_project.src.test_inference import inference_one_cloud
from dianyun.cse.pointcloud_project.src.eval_one_cloud import main as eval_npz_main
from dianyun.cse.pointcloud_project.src.train_pointnet import train as train_main


class PointCloudAPI:
    def __init__(self, project_root, enable_file_receiver=False, client_port=8001, **kwargs):
        # 必需参数
        self.project_root = project_root
        self.enable_file_receiver = enable_file_receiver
        self.client_port = client_port

        self.project_root = os.path.expanduser(project_root)
        self.model_path = "/home/er/MasterComputer/dianyun/cse/pointcloud_project/checkpoints/pointnet_seg_best.pth"

        # 保存最近一次推理中“类别为 2”的点的坐标（N, 3）或 None
        self.last_class2_coords = None

    # ---------------------------
    # 1. 检查环境
    # ---------------------------
    def check_env(self):
        print("🔍 Checking environment...")
        print("Project root:", self.project_root)
        print("CUDA available:", torch.cuda.is_available())
        print("Model path:", self.model_path)

    # ---------------------------
    # 2. C++ 构建（实际上你的 .so 已提供）
    # ---------------------------
    def build_cpp(self):
        print("🔧 C++ backend already built (pc_backend.so). Nothing to do.")

    # ---------------------------
    # 3. 训练
    # ---------------------------
    def train(self):
        print("🚀 Starting training ...")
        train_main()
        print("🎉 Training completed.")

    # ===========================
    # 内部工具函数：
    # 从 inference_one_cloud 的返回中
    # 提取“类别 == 2”的点的坐标
    # ===========================
    def _extract_class2_coords(self, pred_data):
        """
        尝试从推理返回的数据中解析出坐标与标签，并提取标签为 2 的点的坐标。

        兼容几种常见格式：
        1) (points, labels) 的 tuple/list
           - points: (N, 3) 或 (N, >=3)
           - labels: (N,) 或 (N,1)
        2) dict:
           - {"points": ..., "labels": ...}
           - {"coords": ..., "labels": ...}
        3) 单个 ndarray / list:
           - 形状 (N, 4) 或 (N, >=4)，前 3 列为 xyz，最后一列为标签
        """
        if pred_data is None:
            return None

        pts = None
        labels = None

        # 情况 1：字典
        if isinstance(pred_data, dict):
            if "points" in pred_data and "labels" in pred_data:
                pts = np.asarray(pred_data["points"])
                labels = np.asarray(pred_data["labels"])
            elif "coords" in pred_data and "labels" in pred_data:
                pts = np.asarray(pred_data["coords"])
                labels = np.asarray(pred_data["labels"])

        # 情况 2：二元组 / list
        elif isinstance(pred_data, (list, tuple)) and len(pred_data) == 2:
            pts = np.asarray(pred_data[0])
            labels = np.asarray(pred_data[1])

        # 情况 3：直接是一个数组，最后一列为标签
        else:
            arr = np.asarray(pred_data)
            if arr.ndim == 2 and arr.shape[1] >= 4:
                pts = arr[:, :3]
                labels = arr[:, -1].astype(int)

        # 无法解析出 pts / labels
        if pts is None or labels is None:
            return None

        # 保证是 (N,) 形状
        labels = np.asarray(labels).reshape(-1)

        # 过滤出标签为 2 的点
        mask = labels == 2
        if not np.any(mask):
            return None

        pts = np.asarray(pts)
        # 若 pts 不是 (N, 3)，取其前三列作为 xyz
        if pts.ndim == 2 and pts.shape[1] >= 3:
            pts = pts[:, :3]

        return pts[mask]

    # ---------------------------
    # 4. 推理 PLY 点云
    # ---------------------------
    def infer(self, ply_path, out_path, max_print_count=300):
        print("🔎 Running inference on:", ply_path)

        import os
        output_dir = "output_results"
        os.makedirs(output_dir, exist_ok=True)

        filename = os.path.basename(out_path)
        relative_out_path = os.path.join(output_dir, filename)

        # === 新增：兼容 inference_one_cloud 返回 2 个或 3 个值 ===
        infer_result = inference_one_cloud(
            self.model_path, ply_path, relative_out_path
        )

        pred_data = None
        class2_coords = None

        # 如果 inference_one_cloud 返回多个值，进行解包
        if isinstance(infer_result, (list, tuple)):
            if len(infer_result) == 3:
                # 兼容新版： (out_path, pred, xyz_cls2)
                relative_out_path, pred_data, xyz_cls2 = infer_result
                if xyz_cls2 is not None:
                    class2_coords = np.asarray(xyz_cls2)
            elif len(infer_result) == 2:
                # 兼容旧版： (out_path, pred_data)
                relative_out_path, pred_data = infer_result
            else:
                # 非预期长度，尽量保持原有行为
                relative_out_path = infer_result[0]
                if len(infer_result) > 1:
                    pred_data = infer_result[1]
        else:
            # 非 tuple/list，保持原有行为
            relative_out_path = infer_result

        print("🎉 Inference finished. Result saved at:", relative_out_path)

        # 如果模型没有直接返回 xyz_cls2，再尝试从 pred_data 中解析
        if class2_coords is None:
            class2_coords = self._extract_class2_coords(pred_data)

        self.last_class2_coords = class2_coords

        # === 新增：打印类别为 2 的点的坐标（最多 300 个） ===
        if class2_coords is not None:
            num_pts = class2_coords.shape[0]
            print(f"📍 Detected {num_pts} point(s) of class 2. Coordinates:")

            # 限制打印的数量不超过 max_print_count (默认 300)
            print_count = min(num_pts, max_print_count)
            for i, (x, y, z) in enumerate(class2_coords[:print_count]):
                print(f"  #{i:04d}: ({x:.6f}, {y:.6f}, {z:.6f})")

            if num_pts > max_print_count:
                print(f"ℹ️ Only the first {max_print_count} points are printed.")
        else:
            print("ℹ️ No class-2 points detected, or prediction output has no labels/coords.")

        # 返回字典而不是字符串
        result = {
            'result_path': relative_out_path,
            'class2_coordinates': class2_coords.tolist() if class2_coords is not None else None
        }

        return result

    # ---------------------------
    # 5. 评估 npz 点云
    # ---------------------------
    def evaluate(self):
        print("📊 Running evaluation on NPZ dataset ...")
        eval_npz_main()
        print("🎉 Evaluation completed.")
