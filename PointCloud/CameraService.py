import logging
import httpx
import zipfile
import shutil
import os
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import subprocess
from datetime import datetime
from enum import Enum
import aiofiles

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("FileTransfer")

# 调用Python3.9的路径
python_path = "/home/er/.pyenv/shims/python"

# 兼容Pydantic V1和V2
try:
    from pydantic import model_dump

    PYDANTIC_V2 = True
except ImportError:
    PYDANTIC_V2 = False

if not PYDANTIC_V2:
    def model_dump(obj, **kwargs):
        return obj.dict(**kwargs)


# 定义API模型
class ScanConfig(BaseModel):
    camera_ip: Optional[str] = None
    output_dir: Optional[str] = "./scans"
    folder_name: str
    ir_exposure: Optional[int] = 20
    rgb_exposure: Optional[int] = 400
    ir_gain: Optional[int] = 3
    work_mode: Optional[int] = None
    transfer_to_client: bool = False


class BatchScanRequest(BaseModel):
    scan_configs: List[ScanConfig]
    delay_between_scans: Optional[int] = 2


class ScanStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    TRANSFERRING = "transferring"


class ScanResponse(BaseModel):
    scan_id: str
    status: ScanStatus
    success: bool
    message: str
    details: Dict[str, Any]
    timestamp: str
    execution_time: Optional[float] = None
    save_path: Optional[str] = None
    download_url: Optional[str] = None


class BatchScanResponse(BaseModel):
    batch_id: str
    total_scans: int
    completed_scans: int
    failed_scans: int
    scans: List[ScanResponse]
    timestamp: str


class FileTransferRequest(BaseModel):
    scan_id: str
    client_ip: str
    client_port: int = 8001


# 创建FastAPI应用
app = FastAPI(
    title="3D相机扫描API",
    description="用于控制3D相机进行数据采集的REST API",
    version="1.1.0"
)

# 存储扫描状态
scan_results = {}
batch_results = {}
UPLOAD_DIR = "./uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


def fix_path_separators(path: str) -> str:
    """修复路径分隔符，将Windows路径转换为Linux路径"""
    if path:
        # 将反斜杠替换为正斜杠
        path = path.replace('\\', '/')
        # 处理可能的Windows盘符（如C:）
        if ':' in path and len(path) > 2 and path[1] == ':':
            # 如果是绝对路径，去掉盘符，从根目录开始
            path = path[2:]
            if not path.startswith('/'):
                path = '/' + path
    return path


async def transfer_files_to_client(scan_id: str, source_path: str, client_ip: str, client_port: int):
    """将文件传输到客户端 - 使用 httpx 替代 aiohttp"""
    logger.info(
        f"开始传输文件: scan_id={scan_id}, source_path={source_path}, client_ip={client_ip}, client_port={client_port}")

    try:
        # 修复路径分隔符
        source_path = fix_path_separators(source_path)

        # 更新状态为传输中
        if scan_id in scan_results:
            scan_results[scan_id]['status'] = ScanStatus.TRANSFERRING
            scan_results[scan_id]['message'] = '文件传输中'

        # 检查源路径是否存在
        if not os.path.exists(source_path):
            error_msg = f"源路径不存在: {source_path}"
            logger.error(error_msg)
            raise Exception(error_msg)

        # 查找t.bmp和t.ply文件
        t_bmp_path = None
        t_ply_path = None

        for file in os.listdir(source_path):
            if file == "t.bmp":
                t_bmp_path = os.path.join(source_path, file)
            elif file == "t.ply":
                t_ply_path = os.path.join(source_path, file)

        # 检查是否找到两个文件
        if not t_bmp_path or not t_ply_path:
            error_msg = f"在目录中未找到t.bmp和t.ply文件: {source_path}"
            logger.error(error_msg)
            raise Exception(error_msg)

        # 创建临时目录存放要传输的文件
        temp_dir = os.path.join(UPLOAD_DIR, f"transfer_{scan_id}")
        os.makedirs(temp_dir, exist_ok=True)

        # 复制文件到临时目录
        shutil.copy(t_bmp_path, temp_dir)
        shutil.copy(t_ply_path, temp_dir)

        # 压缩临时目录
        zip_path = os.path.join(UPLOAD_DIR, f"{scan_id}.zip")
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file in os.listdir(temp_dir):
                file_path = os.path.join(temp_dir, file)
                zipf.write(file_path, file)

        # 清理临时目录
        shutil.rmtree(temp_dir)

        logger.info(f"创建传输ZIP文件: {zip_path}")

        # 准备传输文件
        client_url = f"http://{client_ip}:{client_port}/receive-file"
        logger.info(f"传输到客户端URL: {client_url}")

        # 使用 httpx 异步客户端发送文件
        async with httpx.AsyncClient() as client:
            # 读取文件内容
            async with aiofiles.open(zip_path, 'rb') as f:
                file_content = await f.read()

            # 构建 multipart 表单数据
            files = {
                'file': (os.path.basename(zip_path), file_content, 'application/zip'),
                'scan_id': (None, scan_id),
                'is_zip': (None, 'true')
            }

            logger.info(f"开始发送文件: {zip_path} (大小: {len(file_content)} bytes)")

            # 发送 POST 请求
            response = await client.post(client_url, files=files)

            if response.status_code == 200:
                result = response.json()
                logger.info(f"文件传输成功: {result}")

                # 更新扫描结果
                if scan_id in scan_results:
                    scan_results[scan_id]['status'] = ScanStatus.COMPLETED
                    scan_results[scan_id]['message'] = '扫描完成且文件已传输'
                    scan_results[scan_id]['download_url'] = result.get('download_url')

                # 清理临时文件
                os.remove(zip_path)
                logger.info(f"删除临时ZIP文件: {zip_path}")

                return True
            else:
                error_msg = f"文件传输失败: HTTP {response.status_code} - {response.text}"
                logger.error(error_msg)
                raise Exception(error_msg)

    except Exception as e:
        error_msg = f'文件传输失败: {str(e)}'
        logger.error(error_msg)

        # 更新错误状态
        if scan_id in scan_results:
            scan_results[scan_id]['status'] = ScanStatus.FAILED
            scan_results[scan_id]['message'] = error_msg

        return False


# 核心扫描函数
def capture_3d_scan(scan_config: ScanConfig, scan_id: str) -> Dict[str, Any]:
    script_path = "Capture.py"

    # 路径分隔符
    final_output_dir = fix_path_separators(scan_config.output_dir or "./scans")
    final_output_path = os.path.join(final_output_dir, scan_config.folder_name)

    cmd = [python_path, script_path]
    cmd.extend(["--output-dir", final_output_dir])
    cmd.extend(["--folder-name", scan_config.folder_name])

    if scan_config.camera_ip:
        cmd.extend(["--camera-ip", scan_config.camera_ip])
    if scan_config.ir_exposure:
        cmd.extend(["--ir-exposure", str(scan_config.ir_exposure)])
    if scan_config.rgb_exposure:
        cmd.extend(["--rgb-exposure", str(scan_config.rgb_exposure)])
    if scan_config.ir_gain:
        cmd.extend(["--ir-gain", str(scan_config.ir_gain)])
    if scan_config.work_mode:
        cmd.extend(["--work-mode", str(scan_config.work_mode)])

    start_time = datetime.now()

    try:
        # 确保目录存在
        os.makedirs(final_output_path, exist_ok=True)
        logger.info(f"创建目录: {final_output_path}")

        # 获取脚本所在目录
        script_dir = os.path.dirname(os.path.abspath(script_path))
        logger.info(f"脚本目录: {script_dir}")
        logger.info(f"执行命令: {' '.join(cmd)}")

        result = subprocess.run(
            cmd,
            capture_output=True,
            encoding='utf-8',
            text=True,
            timeout=120,
            cwd=script_dir
        )

        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()

        success = result.returncode == 0
        output_lines = result.stdout.split('\n')

        # 记录详细的输出信息用于调试
        logger.info(f"命令返回码: {result.returncode}")
        logger.info(f"标准输出: {result.stdout}")
        if result.stderr:
            logger.error(f"错误输出: {result.stderr}")

        scan_info = {
            'success': success,
            'returncode': result.returncode,
            'camera_found': False,
            'camera_connected': False,
            'parameters_set': [],
            'data_captured': False,
            'data_saved': False,
            'save_path': final_output_path,
            'depth_range': None,
            'execution_time': execution_time,
            'stdout': result.stdout,
            'stderr': result.stderr
        }

        actual_save_path = final_output_path
        for line in output_lines:
            if 'camera 0:' in line:
                scan_info['camera_found'] = True
                parts = line.split('camera 0:')
                if len(parts) > 1:
                    scan_info['camera_ip'] = parts[1].split(':')[0].strip()
            elif '成功打开' in line:
                scan_info['camera_connected'] = True
            elif '成功设置参数' in line:
                scan_info['parameters_set'].append(line.strip())
            elif '数据保存到:' in line:
                scan_info['data_saved'] = True
                path_part = line.split('数据保存到:')[-1].strip()
                # 修复保存路径的分隔符
                actual_save_path = fix_path_separators(path_part)
                scan_info['save_path'] = actual_save_path
            elif 'max:' in line and 'min:' in line:
                scan_info['depth_range'] = line.strip()
            elif 'Capture end.' in line:
                scan_info['data_captured'] = True

        return {
            'status': ScanStatus.COMPLETED if success else ScanStatus.FAILED,
            'success': success,
            'message': '扫描完成' if success else '扫描失败',
            'details': scan_info,
            'timestamp': datetime.now().isoformat(),
            'execution_time': execution_time,
            'save_path': actual_save_path
        }

    except subprocess.TimeoutExpired:
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()

        error_msg = '采集超时（超过2分钟）'
        logger.error(error_msg)

        return {
            'status': ScanStatus.TIMEOUT,
            'success': False,
            'message': error_msg,
            'details': {'error': error_msg},
            'timestamp': datetime.now().isoformat(),
            'execution_time': execution_time,
            'save_path': final_output_path
        }
    except Exception as e:
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()

        error_msg = f'执行错误: {str(e)}'
        logger.error(error_msg)

        return {
            'status': ScanStatus.FAILED,
            'success': False,
            'message': error_msg,
            'details': {'error': str(e)},
            'timestamp': datetime.now().isoformat(),
            'execution_time': execution_time,
            'save_path': final_output_path
        }


# API端点
@app.get("/")
async def root():
    return {"message": "3D相机扫描API服务运行中", "status": "active", "version": "1.1.0"}


@app.post("/scan/", response_model=ScanResponse)
async def start_scan(scan_config: ScanConfig, background_tasks: BackgroundTasks):
    """启动单个扫描任务"""
    scan_id = f"scan_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

    # 扫描配置中的路径
    fixed_config = scan_config.copy()
    if fixed_config.output_dir:
        fixed_config.output_dir = fix_path_separators(fixed_config.output_dir)

    # 使用model_dump()
    scan_results[scan_id] = {
        'status': ScanStatus.RUNNING,
        'success': False,
        'message': '扫描进行中',
        'details': {},
        'timestamp': datetime.now().isoformat(),
        'scan_config': model_dump(fixed_config)
    }

    # 执行扫描
    result = capture_3d_scan(fixed_config, scan_id)
    scan_results[scan_id].update(result)

    # 如果扫描成功且需要传输到客户端
    if result['success'] and fixed_config.transfer_to_client:
        scan_results[scan_id]['needs_transfer'] = True
        scan_results[scan_id]['message'] = '扫描完成，等待文件传输'

    return ScanResponse(
        scan_id=scan_id,
        status=result['status'],
        success=result['success'],
        message=result['message'],
        details=result['details'],
        timestamp=result['timestamp'],
        execution_time=result.get('execution_time'),
        save_path=result.get('save_path')
    )


@app.get("/scan/{scan_id}", response_model=ScanResponse)
async def get_scan_status(scan_id: str):
    """获取单个扫描任务的详细状态"""
    if scan_id not in scan_results:
        raise HTTPException(status_code=404, detail="扫描任务不存在")

    result = scan_results[scan_id]

    return ScanResponse(
        scan_id=scan_id,
        status=result['status'],
        success=result['success'],
        message=result['message'],
        details=result['details'],
        timestamp=result['timestamp'],
        execution_time=result.get('execution_time'),
        save_path=result.get('save_path')
    )


@app.post("/transfer-file/")
async def transfer_file(transfer_request: FileTransferRequest, background_tasks: BackgroundTasks):
    """传输文件到客户端"""
    if transfer_request.scan_id not in scan_results:
        raise HTTPException(status_code=404, detail="扫描任务不存在")

    scan_result = scan_results[transfer_request.scan_id]

    if not scan_result['success']:
        raise HTTPException(status_code=400, detail="扫描未成功完成，无法传输文件")

    if 'save_path' not in scan_result or not scan_result['save_path']:
        raise HTTPException(status_code=400, detail="未找到保存路径")

    # 修复保存路径
    fixed_save_path = fix_path_separators(scan_result['save_path'])

    background_tasks.add_task(
        transfer_files_to_client,
        transfer_request.scan_id,
        fixed_save_path,
        transfer_request.client_ip,
        transfer_request.client_port
    )

    return {"message": "文件传输已启动", "scan_id": transfer_request.scan_id}


@app.get("/batch/{batch_id}", response_model=BatchScanResponse)
async def get_batch_status(batch_id: str):
    """获取批量扫描任务状态"""
    if batch_id not in batch_results:
        raise HTTPException(status_code=404, detail="批量扫描任务不存在")

    result = batch_results[batch_id]

    return BatchScanResponse(
        batch_id=batch_id,
        total_scans=result['total_scans'],
        completed_scans=result['completed_scans'],
        failed_scans=result['failed_scans'],
        scans=result['scans'],
        timestamp=result['timestamp']
    )


@app.get("/scans/")
async def list_scans():
    """列出所有扫描任务"""
    return {
        "total_scans": len(scan_results),
        "scans": [
            {
                "scan_id": scan_id,
                "status": result['status'].value if hasattr(result['status'], 'value') else result['status'],
                "success": result['success'],
                "message": result['message']
            }
            for scan_id, result in scan_results.items()
        ]
    }


@app.get("/batches/")
async def list_batches():
    """列出所有批量扫描任务"""
    return {
        "total_batches": len(batch_results),
        "batches": list(batch_results.keys())
    }


# 健康检查端点
@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)