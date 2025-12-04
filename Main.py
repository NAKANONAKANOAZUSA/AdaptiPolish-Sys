import sys
import os
import json

# os.environ['QT_QPA_PLATFORM'] = 'offscreen'

from PyQt5.QtWidgets import QApplication, QMessageBox
from UI.main_window import RobotControlUI

class AppConfig:
    """应用程序配置类"""
    def __init__(self, config_dict):
        # 字符串类型配置
        self.TEACH_POINTS_FILE = str(config_dict.get("TEACH_POINTS_FILE", "teach_points.json"))
        self.ROBOT_IP = str(config_dict.get("ROBOT_IP", "127.0.0.1"))
        self.CALIBRATION_FILE = str(config_dict.get("CALIBRATION_FILE", "calibration_params.json"))
        self.YOLO_MODEL_PATH = str(config_dict.get("YOLO_MODEL_PATH", ""))
        self.OUTPUT_DIR = str(config_dict.get("OUTPUT_DIR", "contour_data"))
        
        # 整数类型配置
        self.ROBOT_PORT = int(config_dict.get("ROBOT_PORT", 9000))
        self.CHUNK = int(config_dict.get("CHUNK", 1024))
        self.FORMAT = int(config_dict.get("FORMAT", 8))
        self.CHANNELS = int(config_dict.get("CHANNELS", 1))
        self.RATE = int(config_dict.get("RATE", 48000))
        self.AUDIO_BUFFER_SIZE = int(config_dict.get("AUDIO_BUFFER_SIZE", 10))
        self.HISTORY_FRAMES = int(config_dict.get("HISTORY_FRAMES", 5))
        self.MIN_CONTOUR_POINTS = int(config_dict.get("MIN_CONTOUR_POINTS", 15))
        self.TARGET_POINTS = int(config_dict.get("TARGET_POINTS", 200))
        
        # 浮点数类型配置
        self.CONFIDENCE_THRESHOLD = float(config_dict.get("CONFIDENCE_THRESHOLD", 0.9))
        self.SMOOTH_SIGMA = float(config_dict.get("SMOOTH_SIGMA", 1.0))
    
    def __repr__(self):
        """返回配置的字符串表示"""
        return "\n".join(f"{key}: {value}" for key, value in vars(self).items())

def load_config(file_path="config.json"):
    """
    从JSON文件加载配置
    
    :param file_path: 配置文件路径
    :return: AppConfig实例
    """
    # 如果配置文件不存在，使用默认配置
    if not os.path.exists(file_path):
        print(f"警告: 配置文件 {file_path} 不存在，使用默认配置")
        return AppConfig({})
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
        return AppConfig(config_data)
    
    except json.JSONDecodeError as e:
        print(f"配置文件格式错误: {str(e)}，使用默认配置")
        return AppConfig({})
    except Exception as e:
        print(f"加载配置时出错: {str(e)}，使用默认配置")
        return AppConfig({})

def show_config_warning():
    """显示配置加载警告对话框"""
    msg = QMessageBox()
    msg.setIcon(QMessageBox.Warning)
    msg.setWindowTitle("配置加载警告")
    msg.setText("配置文件加载失败或不存在，已使用默认配置")
    msg.setInformativeText("请检查 config.json 文件是否存在且格式正确")
    msg.setStandardButtons(QMessageBox.Ok)
    msg.exec_()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # 加载配置文件
    config = load_config()
    
    # 如果配置文件不存在或格式错误，显示警告
    if not os.path.exists("config.json"):
        show_config_warning()
    
    # 创建主窗口并传递配置
    window = RobotControlUI(config)
    
    # 设置窗口大小
    screen_rect = app.primaryScreen().availableGeometry()
    window.setMaximumSize(screen_rect.width(), screen_rect.height())
    
    # 显示窗口
    window.show()
    
    # 退出应用
    sys.exit(app.exec_())