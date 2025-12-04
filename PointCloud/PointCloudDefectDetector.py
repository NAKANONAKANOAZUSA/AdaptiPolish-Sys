import glob
import logging
import os
import json
import re
import socket
from sqlite3.dbapi2 import apilevel
import logging
import threading
import time
import zipfile
from datetime import datetime
import aiofiles

import requests
import numpy as np
from dianyun.cse.pointcloud_project.src.sss_API import PointCloudAPI
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, BackgroundTasks
import uvicorn
# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("FileReceiver")

# 客户端文件接收服务
client_app = FastAPI(title="3D扫描客户端文件接收服务")

# 客户端存储目录
CLIENT_STORAGE = "./client_scans"
os.makedirs(CLIENT_STORAGE, exist_ok=True)


@client_app.post("/receive-file")
async def receive_file(
        background_tasks: BackgroundTasks,
        scan_id: str = Form(...),
        is_zip: str = Form(...),
        file: UploadFile = File(...)
):
    """接收从服务器传输的文件"""
    logger.info(f"接收到文件传输请求: scan_id={scan_id}, is_zip={is_zip}, filename={file.filename}")

    try:
        scan_dir = os.path.join(CLIENT_STORAGE, scan_id)
        os.makedirs(scan_dir, exist_ok=True)
        logger.info(f"创建扫描目录: {scan_dir}")

        is_zip_bool = is_zip.lower() == 'true'
        filename = file.filename or f"{scan_id}.file"

        if is_zip_bool:
            logger.info(f"接收ZIP文件: {filename}")
            zip_path = os.path.join(scan_dir, filename)

            # 保存ZIP文件
            file_content = await file.read()
            async with aiofiles.open(zip_path, 'wb') as f:
                await f.write(file_content)
            logger.info(f"ZIP文件保存成功: {zip_path} (大小: {len(file_content)} bytes)")

            # 在后台任务中解压ZIP文件
            background_tasks.add_task(extract_zip_file, zip_path, scan_dir)

            return {
                "status": "success",
                "message": "文件接收成功，正在解压",
                "scan_id": scan_id,
                "save_path": scan_dir
            }
        else:
            logger.info(f"接收单个文件: {filename}")
            file_path = os.path.join(scan_dir, filename)

            # 保存文件
            file_content = await file.read()
            async with aiofiles.open(file_path, 'wb') as f:
                await f.write(file_content)
            logger.info(f"文件保存成功: {file_path} (大小: {len(file_content)} bytes)")

            return {
                "status": "success",
                "message": "文件接收完成",
                "scan_id": scan_id,
                "save_path": file_path
            }

    except Exception as e:
        error_msg = f"文件接收失败: {str(e)}"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)


async def extract_zip_file(zip_path: str, extract_dir: str):
    """在后台解压ZIP文件"""
    try:
        logger.info(f"开始解压ZIP文件: {zip_path} -> {extract_dir}")

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)

        logger.info(f"解压完成: {extract_dir}")

        # 删除ZIP文件
        os.remove(zip_path)
        logger.info(f"删除ZIP文件: {zip_path}")

    except Exception as e:
        logger.error(f"解压ZIP文件失败: {str(e)}")


@client_app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@client_app.get("/files/{scan_id}")
async def list_files(scan_id: str):
    scan_dir = os.path.join(CLIENT_STORAGE, scan_id)
    if not os.path.exists(scan_dir):
        raise HTTPException(status_code=404, detail="扫描文件不存在")

    files = []
    for root, dirs, filenames in os.walk(scan_dir):
        for filename in filenames:
            file_path = os.path.join(root, filename)
            relative_path = os.path.relpath(file_path, scan_dir)
            files.append({
                "name": filename,
                "path": relative_path,
                "size": os.path.getsize(file_path)
            })

    return {
        "scan_id": scan_id,
        "files": files,
        "total_files": len(files)
    }


def get_local_ip():
    """获取本机IP地址"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "127.0.0.1"


def start_client_server(port=8001):
    """启动客户端文件接收服务器"""

    def run_server():
        uvicorn.run(client_app, host="0.0.0.0", port=port, log_level="info")

    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    return server_thread


class PointCloudDefectDetector:
    def __init__(self,
                 defect_api_url="http://192.168.25.184:9000",
                 camera_api_url="http://192.168.25.184:8000",
                 standard_part_path=r"E:\pointcloud_ai_project2\data\npy\standard_part.npy",
                 client_port=8001,
                 start_file_receiver=True):
        """
        点云缺陷检测器

        Args:
            defect_api_url: 缺陷检测API地址（服务端IP）
            camera_api_url: 相机API地址（服务端IP）
            standard_part_path: 标准工件文件路径（服务端Windows路径）
            client_port: 客户端文件接收端口
            start_file_receiver: 是否启动文件接收服务
        """
        self.defect_api_url = defect_api_url
        self.camera_api_url = camera_api_url
        self.standard_part_path = standard_part_path
        self.client_port = client_port
        self.client_storage = CLIENT_STORAGE

        # 初始化PointCloudAPI（文档3的功能）
        self.pointcloud_api = PointCloudAPI("dianyun/cse/pointcloud_project")

        # 启动客户端文件接收服务
        if start_file_receiver:
            print("启动客户端文件接收服务...")
            self.server_thread = start_client_server(client_port)
            time.sleep(2)  # 等待服务器启动
            print(f"客户端文件接收服务已启动，端口: {client_port}")

    def get_client_ip(self):
        """获取客户端IP地址"""
        return get_local_ip()

    def request_scan(self, transfer_to_client=True):
        """请求相机扫描并返回文件路径"""
        client_ip = self.get_client_ip()
        print(f"客户端IP: {client_ip}")

        # 扫描配置
        scan_config = {
            "camera_ip": None,
            "output_dir": "./scans",
            "folder_name": "test_object",
            "ir_exposure": 20,
            "rgb_exposure": 400,
            "ir_gain": 3,
            "transfer_to_client": transfer_to_client
        }

        try:
            # 发送扫描请求
            print("开始扫描...")
            response = requests.post(f"{self.camera_api_url}/scan/", json=scan_config, timeout=300)
            result = response.json()
            print("扫描结果:")
            print(json.dumps(result, indent=2, ensure_ascii=False))

            scan_id = result.get('scan_id')
            save_path = result.get('save_path')  # 获取文件保存路径

            if result.get('success'):
                # 如果要求传输到客户端
                if transfer_to_client:
                    # 请求文件传输
                    print("请求文件传输...")
                    transfer_request = {
                        "scan_id": scan_id,
                        "client_ip": client_ip,
                        "client_port": self.client_port
                    }

                    transfer_response = requests.post(
                        f"{self.camera_api_url}/transfer-file/",
                        json=transfer_request,
                        timeout=300
                    )
                    transfer_result = transfer_response.json()
                    print("传输响应:", transfer_result)

                    # 等待文件传输完成
                    time.sleep(5)  # 给文件传输和解压留出时间

                    return {
                        "success": True,
                        "scan_id": scan_id,
                        "server_path": save_path,
                        "client_path": os.path.join(self.client_storage, scan_id),
                        "transferred_to_client": True,
                        "message": "扫描完成且文件已传输到客户端"
                    }
                else:
                    # 不传输到客户端
                    return {
                        "success": True,
                        "scan_id": scan_id,
                        "server_path": save_path,
                        "transferred_to_client": False,
                        "message": f"扫描完成！文件保存在服务器: {save_path}"
                    }
            else:
                return {
                    "success": False,
                    "message": f"扫描失败: {result.get('message')}",
                    "scan_id": scan_id
                }

        except requests.exceptions.RequestException as e:
            return {
                "success": False,
                "message": f"请求失败: {e}"
            }

    def get_latest_scan_file(self):
        """获取最新的扫描文件路径"""
        # 获取所有scan_开头的文件夹
        scan_folders = glob.glob(os.path.join(self.client_storage, "scan_*"))

        if not scan_folders:
            print("未找到扫描文件夹")
            return None

        # 提取时间戳并排序（格式：scan_20251203_203029_623386）
        def extract_timestamp(folder_path):
            folder_name = os.path.basename(folder_path)
            match = re.match(r'scan_(\d{8})_(\d{6})_(\d+)', folder_name)
            if match:
                date_str, time_str, micro_str = match.groups()
                return (date_str, time_str, micro_str)
            return ("00000000", "000000", "000000")

        # 按时间戳降序排列（最新的在前面）
        scan_folders.sort(key=extract_timestamp, reverse=True)
        latest_scan_folder = scan_folders[0]

        t_ply_path = os.path.join(latest_scan_folder, "t.ply")

        # 检查文件是否存在
        if os.path.exists(t_ply_path):
            print(f"找到最新扫描文件: {t_ply_path}")
            return t_ply_path
        else:
            print(f"文件不存在: {t_ply_path}")
            # 尝试寻找其他可能的点云文件
            ply_files = glob.glob(os.path.join(latest_scan_folder, "*.ply"))
            if ply_files:
                print(f"使用替代文件: {ply_files[0]}")
                return ply_files[0]
            return None

    def detect_defects(self, ply_path=None):
        """
        使用文档3的PointCloudAPI检测缺陷点

        Args:
            ply_path: 点云文件路径，如果为None则自动查找最新文件
        """
        # 如果没有指定文件路径，自动查找最新扫描文件
        if ply_path is None:
            ply_path = self.get_latest_scan_file()
            if ply_path is None:
                print("未找到可用的点云文件")
                return None

        print(f"使用点云文件进行缺陷检测: {ply_path}")

        try:
            # 调用文档3的PointCloudAPI进行推理
            detection_result = self.pointcloud_api.infer(ply_path, "result.ply", 200)

            # 确保返回的结果包含必要的字段
            if detection_result is None:
                detection_result = {}

            # 添加文件路径信息
            detection_result['pointcloud_file'] = ply_path

            # 处理类别2的坐标点（缺陷点）
            class2_coords = detection_result.get('class2_coordinates')
            if class2_coords is not None:
                # 转换格式为缺陷点列表
                defect_points = []
                for i, coords in enumerate(class2_coords):
                    if len(coords) >= 3:
                        defect_points.append({
                            'x': coords[0],
                            'y': coords[1],
                            'z': coords[2],
                            'distance': 0.0,  # 可以根据需要计算实际距离
                            'point_id': i
                        })

                detection_result['defect_points'] = defect_points
                detection_result['num_defects'] = len(defect_points)
            else:
                detection_result['defect_points'] = []
                detection_result['num_defects'] = 0

            detection_result.setdefault('unit', 'mm')
            detection_result.setdefault('transform_matrix', [])

            return detection_result

        except Exception as e:
            print(f"缺陷检测过程中发生错误: {e}")
            return None

    def get_3d_coordinates(self):
        """完整流程：扫描 → 传输 → 解压 → 检测 → 返回缺陷点坐标"""
        # 1. 请求相机扫描（传输文件到客户端）
        print("步骤1: 请求相机扫描并传输文件到客户端...")
        scan_result = self.request_scan(transfer_to_client=True)

        if not scan_result['success']:
            return {
                "success": False,
                "message": f"扫描失败: {scan_result.get('message')}",
                "scan_id": scan_result.get('scan_id')
            }

        # 2. 等待文件传输和解压完成
        print("步骤2: 等待文件传输和解压...")
        time.sleep(3)

        # 3. 使用文档3的API进行缺陷检测
        print("步骤3: 调用缺陷检测API...")
        detection_result = self.detect_defects()

        if not detection_result:
            return {
                "success": False,
                "message": "缺陷检测失败",
                "scan_id": scan_result['scan_id'],
                "pointcloud_file": None
            }

        # 4. 返回结果
        print("步骤4: 返回检测结果...")
        return {
            "success": True,
            "scan_id": scan_result['scan_id'],
            "pointcloud_file": detection_result.get('pointcloud_file'),
            "defect_points": detection_result.get('defect_points', []),
            "num_defects": detection_result.get('num_defects', 0),
            "unit": detection_result.get('unit', 'mm'),
            "transform_matrix": detection_result.get('transform_matrix', []),
            "class2_coordinates": detection_result.get('class2_coordinates'),
            "message": f"检测完成，发现 {detection_result.get('num_defects', 0)} 个缺陷点"
        }


# 使用示例
if __name__ == "__main__":
    # 创建检测器实例 - 会自动启动文件接收服务
    detector = PointCloudDefectDetector(
        defect_api_url="http://192.168.25.184:9000",
        camera_api_url="http://192.168.25.184:8000",
        standard_part_path=r"E:\pointcloud_ai_project2\data\npy\standard_part.npy",
        client_port=8001,
        start_file_receiver=True
    )

    # 执行完整流程
    result = detector.get_3d_coordinates()

    print("\n" + "=" * 50)
    print("最终结果:")
    print("=" * 50)

    if result['success']:
        print(f"✓ 操作成功!")
        print(f"扫描ID: {result['scan_id']}")
        print(f"点云文件: {result['pointcloud_file']}")
        print(f"发现缺陷点数量: {result['num_defects']}")
        print(f"单位: {result['unit']}")

        # 打印缺陷点坐标
        if result['defect_points']:
            print("\n缺陷点坐标:")
            for i, point in enumerate(result['defect_points'][:10]):  # 只显示前10个
                print(f"缺陷点 {i + 1}: ({point['x']:.3f}, {point['y']:.3f}, {point['z']:.3f})")

        # 也可以直接显示类别2的坐标
        if result.get('class2_coordinates'):
            print(f"\n类别2坐标点数量: {len(result['class2_coordinates'])}")
    else:
        print(f"✗ 操作失败: {result['message']}")

    # 保持服务运行
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("服务已停止")