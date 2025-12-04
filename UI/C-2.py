from datetime import datetime
import requests  # 添加这行导入
import json
import socket
import time  # 添加这行导入
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, BackgroundTasks
import aiofiles
import uvicorn
import zipfile
import logging
import os

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


# 添加健康检查端点
@client_app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


# 添加文件列表端点
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


# 获取本机IP地址
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


# 启动客户端文件接收服务器
def start_client_server(port=8001):
    """启动客户端文件接收服务器"""
    import threading

    def run_server():
        uvicorn.run(client_app, host="0.0.0.0", port=port, log_level="info")

    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    return server_thread


# 扫描请求函数
def request_scan(transfer_to_client=True):
    """请求扫描并处理文件传输"""
    client_ip = get_local_ip()
    print(f"客户端IP: {client_ip}")

    # 扫描配置
    scan_config = {
        "camera_ip": None,
        "output_dir": "./scans",
        "folder_name": "test_object",
        "ir_exposure": 20,
        "rgb_exposure": 400,
        "ir_gain": 3,
        "transfer_to_client": transfer_to_client  # 控制是否传输文件
    }

    try:
        # 发送扫描请求
        print("开始扫描...")
        response = requests.post("http://192.168.25.184:8000/scan/", json=scan_config, timeout=300)
        result = response.json()
        print("扫描结果:")
        print(json.dumps(result, indent=2, ensure_ascii=False))

        scan_id = result.get('scan_id')

        if result.get('success'):
            if transfer_to_client:
                # 请求文件传输
                print("请求文件传输...")
                transfer_request = {
                    "scan_id": scan_id,
                    "client_ip": client_ip,
                    "client_port": 8001
                }

                transfer_response = requests.post(
                    "http://192.168.25.184:8000/transfer-file/",
                    json=transfer_request,
                    timeout=300
                )
                print("传输响应:", transfer_response.json())
            else:
                print(f"扫描完成！文件保存在服务器: {result.get('save_path')}")
        else:
            print(f"扫描失败: {result.get('message')}")

    except requests.exceptions.RequestException as e:
        print(f"请求失败: {e}")


if __name__ == "__main__":
    # 启动客户端文件接收服务
    print("启动客户端文件接收服务...")
    start_client_server(8001)

    # 等待服务器启动
    time.sleep(2)

    # 测试扫描（传输文件到客户端）
    print("\n=== 测试扫描（传输文件到客户端）===")
    request_scan(transfer_to_client=True)


    # 保持客户端运行
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("客户端服务已停止")