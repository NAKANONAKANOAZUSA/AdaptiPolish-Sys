import sys
import time
import cv2
import numpy as np
import socket
import struct
from PyQt5.QtCore import QThread, pyqtSignal
class CameraThread(QThread):
    """摄像头线程"""
    update_frame = pyqtSignal(np.ndarray)
    status_changed = pyqtSignal(str)

    def __init__(self, config,camera_type="local", ip=None, port=None, parent=None):
        super().__init__(parent)
        self.config=config
        self.camera_type = camera_type
        self.ip = ip
        self.port = port
        self.running = False
        self.cap = None
        self.server_socket = None
        self.client_socket = None

    def run(self):
        """启动摄像头流"""
        self.running = True

        if self.camera_type == "network":
            self.status_changed.emit("正在连接网络摄像头...")
            try:
                self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.server_socket.bind((self.ip, self.port))
                self.server_socket.listen(5)
                self.status_changed.emit(f"等待摄像头客户端连接: {self.ip}:{self.port}")

                self.client_socket, addr = self.server_socket.accept()
                self.status_changed.emit(f"摄像头客户端已连接: {addr[0]}:{addr[1]}")
                payload_size = struct.calcsize('>I')
                data = b""

                while self.running:
                    try:
                        # 读取帧长度信息
                        while len(data) < payload_size:
                            packet = self.client_socket.recv(4096)
                            if not packet:
                                break
                            data += packet

                        if len(data) < payload_size:
                            break

                        packed_msg_size = data[:payload_size]
                        data = data[payload_size:]
                        msg_size = struct.unpack('>I', packed_msg_size)[0]

                        # 读取完整的帧数据
                        while len(data) < msg_size:
                            data += self.client_socket.recv(4096)

                        frame_data = data[:msg_size]
                        data = data[msg_size:]
                        frame_array = np.frombuffer(frame_data, dtype=np.uint8)
                        frame = cv2.imdecode(frame_array, flags=cv2.IMREAD_COLOR)

                        if frame is not None:
                            self.update_frame.emit(frame)
                    except Exception as e:
                        self.status_changed.emit(f"摄像头接收错误: {str(e)}")
                        break
            except Exception as e:
                self.status_changed.emit(f"启动网络摄像头失败: {str(e)}")
            finally:
                self.cleanup()
        else:
            # 本地摄像头
            self.status_changed.emit("正在打开本地摄像头...")
            try:
                # 根据操作系统选择不同的后端
                if sys.platform.startswith('linux'):
                    self.cap = cv2.VideoCapture(20, cv2.CAP_V4L2)
                elif sys.platform.startswith('win'):
                    self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
                else:
                    self.cap = cv2.VideoCapture(20)

                if not self.cap.isOpened():
                    self.status_changed.emit("无法打开本地摄像头")
                    return

                self.status_changed.emit("本地摄像头已启动")

                while self.running:
                    ret, frame = self.cap.read()
                    if ret:
                        self.update_frame.emit(frame)
                    else:
                        self.status_changed.emit("无法从摄像头读取帧")
                    time.sleep(0.03)
            except Exception as e:
                self.status_changed.emit(f"本地摄像头错误: {str(e)}")
            finally:
                self.cleanup()

    def cleanup(self):
        """清理资源"""
        if self.cap and self.cap.isOpened():
            self.cap.release()
        if self.client_socket:
            self.client_socket.close()
        if self.server_socket:
            self.server_socket.close()
        self.cap = None
        self.server_socket = None
        self.client_socket = None

    def stop(self):
        """停止摄像头"""
        self.running = False
        self.wait()