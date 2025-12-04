import threading
import time
from queue import Queue
from pymycobot import MyCobot280Socket, MyCobot280
class RobotConnection:
    """机械臂连接管理器"""

    def __init__(self,config,ip=None ,port=None):
        self.config = config
        self.ip = ip                    # 机器人IP地址
        self.port = port                # 机器人端口号
        self.mc = None                  # 机器人连接对象
        self.connected = False          # 连接状态标志
        self.heartbeat_active = True    # 心跳检测活动状态
        self.heartbeat_thread = None    # 心跳检测线程对象
        self.reconnect_queue = Queue()  # 重连队列
        self.lock = threading.Lock()    # 线程锁
        self.connect()                  # 初始化时立即尝试连接
        self.start_heartbeat()          # 启动心跳检测线程
        self.reconnect_lock = threading.Lock()  # 重连锁
        self.reconnecting = False       # 重连状态标志

    def connect(self):
        """尝试连接机械臂"""
        try:
            with self.lock:
                if self.mc:
                    try:
                        self.mc.close()
                    except:
                        pass

                # self.mc = MyCobot280Socket(self.ip, self.port)
                self.mc = MyCobot280("/dev/ttyAMA0", 1000000)
                test_result = self.mc.is_controller_connected()

                if test_result in [0, 1]:
                    self.connected = True
                    print(f"成功连接到机械臂: {self.ip}:{self.port}")
                    return True
                else:
                    print(f"连接测试失败: {test_result}")
                    return False
        except Exception as e:
            print(f"连接失败: {str(e)}")
            self.connected = False
            return False

    def disconnect(self):
        """断开连接"""
        try:
            with self.lock:
                if self.mc:
                    self.mc.close()
                self.connected = False
                print("已断开机械臂连接")
                return True
        except:
            return False

    def reconnect(self):
        """重新连接方法"""
        with self.reconnect_lock:  # 获取重连锁
            if self.reconnecting:
                print("重连操作已在其他线程进行")
                return False
                
            try:
                self.reconnecting = True
                # 重连
                self.disconnect()
                time.sleep(0.5)
                result = self.connect()
                return result
            finally:
                self.reconnecting = False

    def start_heartbeat(self):
        """启动心跳检测线程"""
        if not self.heartbeat_thread or not self.heartbeat_thread.is_alive():
            self.heartbeat_active = True
            self.heartbeat_thread = threading.Thread(target=self.heartbeat_check, daemon=False)
            self.heartbeat_thread.start()
            print("心跳检测已启动")

    def stop_heartbeat(self):
        """停止心跳检测"""
        self.heartbeat_active = False
        if self.heartbeat_thread and self.heartbeat_thread.is_alive():
            self.heartbeat_thread.join(timeout=2.0)
        print("心跳检测已停止")

    def heartbeat_check(self):
        """心跳检测循环"""
        while self.heartbeat_active:
            time.sleep(5)
            
            # 如果正在重连则跳过
            if self.reconnecting:
                continue
                
            try:
                # 检查连接状态
                if not self.mc:
                    self.connected = False
                    print("机械臂对象未初始化")
                    continue
                    
                status = self.mc.is_controller_connected()
                if status != 1:
                    print(f"心跳检测失败: {status}")
                    self.connected = False
                    
                    # 使用带锁的重连方法
                    if self.reconnect_lock.acquire(blocking=False):
                        try:
                            print("开始重新连接...")
                            self.reconnect()
                        finally:
                            self.reconnect_lock.release()
                    else:
                        print("重连操作已由其他线程启动")
                else:
                    self.connected = True
            except Exception as e:
                print(f"心跳检测异常: {str(e)}")
                self.connected = False

    def is_connected(self):
        """返回连接状态"""
        return self.connected

    def get_robot(self):
        """获取机械臂对象，确保线程安全"""
        if not self.connected:
            print("等待重新连接...")
            if not self.reconnect_queue.get(timeout=30):
                raise ConnectionError("无法重新连接机械臂")
        with self.lock:
            return self.mc