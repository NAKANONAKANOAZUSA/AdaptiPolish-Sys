import queue
import socket
from PyQt5.QtCore import QThread, pyqtSignal
class AudioServerThread(QThread):
    """麦克风服务器线程"""
    status_changed = pyqtSignal(str)
    audio_data_ready = pyqtSignal(bytes)

    def __init__(self, config, parent=None):
        super().__init__(parent)
        self.config = config
        self.running = False
        self.server_socket = None
        self.connection = None
        self.audio_queue = queue.Queue(maxsize=self.config.AUDIO_BUFFER_SIZE)
        self.audio_queue = queue.Queue(self.config.AUDIO_BUFFER_SIZE)

    def run(self):
        """启动音频服务器"""
        try:
            self.status_changed.emit("正在启动麦克风服务器...")

            # 创建TCP Socket
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.bind(('0.0.0.0', 12345))
            self.server_socket.listen(1)
            self.server_socket.settimeout(5)  # 设置超时时间

            self.status_changed.emit("等待麦克风客户端连接...")
            self.running = True
            self.status_changed.emit("麦克风服务器已启动，等待连接")

            # 接受连接
            self.connection, addr = self.server_socket.accept()
            self.status_changed.emit(f"麦克风客户端已连接: {addr[0]}:{addr[1]}")
            print("开始接收音频数据...")

            # 音频接收循环
            while self.running:
                try:
                    # 接收数据并存储
                    data = self.connection.recv(self.config.CHUNK)
                    if not data:
                        self.status_changed.emit("麦克风连接断开")
                        break

                    # 将数据放入队列
                    if self.audio_queue.full():
                        self.audio_queue.get() 
                    self.audio_queue.put(data)

                    # 发出信号通知有新音频数据
                    self.audio_data_ready.emit(data)

                except (ConnectionResetError, BrokenPipeError):
                    self.status_changed.emit("麦克风连接异常中断")
                    break
                except socket.timeout:
                    # 超时但连接未断开，继续等待
                    continue
                except Exception as e:
                    self.status_changed.emit(f"音频接收错误: {str(e)}")
                    break

        except socket.timeout:
            self.status_changed.emit("麦克风连接超时")
        except Exception as e:
            self.status_changed.emit(f"启动麦克风服务器失败: {str(e)}")
        finally:
            self.cleanup()

    def get_audio_data(self):
        """获取最新的音频数据块"""
        try:
            return self.audio_queue.get_nowait()
        except queue.Empty:
            return None

    def get_all_audio_data(self):
        """获取所有缓冲的音频数据"""
        data = b""
        while not self.audio_queue.empty():
            data += self.audio_queue.get_nowait()
        return data

    def stop(self):
        """停止音频服务器"""
        self.running = False
        if self.connection:
            try:
                self.connection.shutdown(socket.SHUT_RDWR)
                self.connection.close()
            except:
                pass
        self.cleanup()

    def cleanup(self):
        """清理资源"""
        if self.server_socket:
            try:
                self.server_socket.close()
            except:
                pass

        # 清空音频队列
        while not self.audio_queue.empty():
            self.audio_queue.get_nowait()

        self.connection = None
        self.server_socket = None