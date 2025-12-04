import torch
from dianyun.cse.pointcloud_project.src.pc_backend import load_pointcloud, save_colored_ply   # C++

from dianyun.cse.pointcloud_project.src.model_pointnet import SimplePointNetSeg

def inference_one_cloud(model_path, ply_path, out_path="infer_result.ply"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 加载模型
    model = SimplePointNetSeg(num_classes=3).to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # 2. 用 C++ 加载 PLY
    xyz = load_pointcloud(ply_path)        # numpy (N,3)
    xyz_t = torch.from_numpy(xyz).float().unsqueeze(0).to(device)

    # 3. 推理
    with torch.no_grad():
        logits = model(xyz_t.transpose(1, 2))   # (1,N,3)
        pred = logits.argmax(dim=-1).squeeze(0).cpu().numpy()

    # 4. 保存上色结果
    save_colored_ply(out_path, xyz, pred)

    return out_path, pred
