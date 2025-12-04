import asyncio
import os
import socket
import re
import time
import json
import math
import threading
import cv2
import cn2an
import numpy as np

from pypinyin import lazy_pinyin, Style
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QLineEdit, QPushButton, QGroupBox,
                             QListWidget, QListWidgetItem, QAbstractItemView,
                             QMessageBox, QInputDialog, QProgressBar,
                             QComboBox, QSlider, QGridLayout,
                             QMainWindow, QFileDialog, QSplitter, QFrame, QCheckBox, QScrollArea)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont, QColor, QBrush, QImage, QPixmap, QIcon, QDoubleValidator, QIntValidator
from hardware.robot_basic import RobotConnection
from hardware.motor import MotorController
from vision.ONNXDetectionThread import ONNXDetectionThread
from vision.camera_thread import CameraThread
from vision.contour_manager import FixedContourManager
from vision.CameraDetectionSystem import CameraDetectionSystem
from audio.audio_server import AudioServerThread
from audio.audio_system import AudioSystem
from audio.MicrophoneCalibrator import MicrophoneCalibrator
from audio.RecordingThread import RecordingThread
from UI.manual_control_dialog import MotorControlDialog
from UI.UART.lib.TOF_Sense import TOF_Sense
from UI.ManualControlDialog import ManualControlDialog
from UI.ThreadPoolManager import  ThreadPoolManager
from Polish.MyCobotGrindingController import MyCobotGrindingController

class RobotControlUI(QMainWindow):
    """机器人控制主界面"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        # self.audio_server = AudioServerThread(config=config,parent=self)
        self.audio_system = AudioSystem(config=config)
        # self.motor = MotorController()
        # self.robot_basic = RobotConnection(config=config,port=None,ip=None)
        # self.manual_control_dialog = MotorControlDialog()
        self.camera_thread = CameraThread(config=config)
        # self.contour_manager = FixedContourManager(config=config)
        # self.ONNXDetectionThread = ONNXDetectionThread(config=config,ip='0.0.0.0', port=9999)
        self.CameraDetectionSystem = CameraDetectionSystem(config=config,server_ip='0.0.0.0', server_port=9999)
        self.grinding_controller = None
        self.current_grinding_task_id = None
        # 创建线程池管理器
        self.thread_pool = ThreadPoolManager(max_workers=5)
        self.thread_pool.task_completed.connect(self.handle_task_completed)
        self.thread_pool.task_failed.connect(self.handle_task_failed)

        # 任务ID映射，用于跟踪特定类型的任务
        self.task_ids = {
            'camera': None,
            'detection': None,
            'audio': None,
            'motion': None,
            'calibration': None,
            'grinding': None
        }

        # ===== 窗口设置 =====
        self.setWindowTitle("机器人控制系统")       # 设置窗口标题
        self.setGeometry(100, 100, 1200, 800)      # 设置窗口位置和大小
        self.setMinimumSize(800, 600)              # 设置最小窗口尺寸
        self.setWindowIcon(QIcon("Icon.ico"))      # 设置窗口图标

        # ===== 执行控制相关 =====
        self.execution_paused = False              # 执行暂停标志
        self.execution_stopped = False             # 执行停止标志
        self.execution_thread = None               # 执行线程
        self.execution_progress = 0                # 执行进度
        self.verification_enabled = True           # 位置验证启用标志
        self.angle_tolerance = 3.0                 # 角度容差（度）
        self.coord_tolerance = 5.0                 # 坐标容差（毫米）

        # ===== 检测与打磨参数 =====
        self.detection_thread = None               # 目标检测线程
        self.detection_active = False              # 检测活动状态
        self.grinding_loops = 1                    # 打磨循环次数
        self.grinding_x_step = 0.0                 # X方向进深（毫米）
        self.grinding_y_step = 0.0                 # Y方向进深（毫米）
        self.grinding_z_step = 0.0                 # Z方向进深（毫米）
        self.path_scale_factor_X = 1.0             # 路径缩放比例X
        self.path_scale_factor_Y = 1.0             # 路径缩放比例Y
        self.grinding_current_loop = 0             # 当前打磨循环次数
        self.base_distance = 245                   # 基准距离245mm
        self.base_scale_x = 0.67                 # 基准X缩放
        self.base_scale_y = 0.55                   # 基准Y缩放
        self.current_distance = 245                # 当前距离（默认基准值）

        self.history_paths = []                    # 存储历史路径
        self.current_history_path = None           # 当前选择的历史路径
        self.use_history_path = False              # 是否使用历史路径的标志
        self.coordinate_rotation = 270

        # ===== 示教点管理 =====
        self.teach_points = []                     # 存储示教点列表
        self.current_point = None                  # 当前选中的示教点
        self.last_executed_point = None            # 最后执行的示教点
        self.load_teach_points()                   # 加载保存的示教点

        # ===== 摄像头与模型 =====
        self.camera_type = "local"                 # 摄像头类型（local/network）
        self.onnx_model_path = ""                  # ONNX模型路径

        # 语音识别相关
        self.microphone_btn = None
        self.mic_status_label = None
        self.speech_recognition_btn = None
        self.calibration_recognition_btn = None
        self.load_calibration_recognition_btn= None
        self.speech_recognition_active = False  # 语音识别激活状态
        self.recording_thread = None
        self.is_recording = False
        self.audio_threshold = 0.02  # 默认阈值
        self.background_level = 0.01  # 背景噪音水平
        self.speech_level = 0.05     # 正常说话水平
        self.calibrator = None        # 麦克风校准器

        # 唤醒词相关状态
        self.wake_word = "小智"  # 唤醒词
        self.is_waiting_for_wake_word = False  # 是否在等待唤醒词
        self.is_in_command_mode = False  # 是否在指令模式
        self.wake_word_detected = False  # 是否检测到唤醒词

        # 定时器用于控制录音时长
        self.recording_timer = QTimer(self)
        self.recording_timer.timeout.connect(self.stop_recording_for_processing)

        # 唤醒词检测线程
        self.wake_word_thread = None

        # 初始化UI
        self.init_ui()                             # 初始化用户界面




        # ===== 机器人连接与状态 =====
        self.connection = None                     # 机器人连接对象
        self.mc = None                             # 机器人控制对象
        self.update_ui_state(False)                # 初始化UI状态

        # ===== 用户偏移量 =====
        self.user_offset_x = 0                     # 用户自定义X偏移量
        self.user_offset_y = 0                     # 用户自定义Y偏移量
        self.user_offset_z = 0                     # 用户自定义Z偏移量

        # ===== 音频与语音识别 =====
        self.audio_thread = None                   # 音频处理线程
        self.speech_recognition_active = False     # 语音识别激活状态

        # from llm import LlmModel
        # self.llm_model = LlmModel()
        self.llm_model = None
        #
        # from asr import AsrModel
        # self.asr_model = AsrModel()  # 语音识别模型
        self.asr_model = None
        # # 创建TTS模型实例
        # from tts import TtsModel
        # self.tts_model = TtsModel()
        self.tts_model = None

        # self.audio_system = AudioSystem(self.tts_model)
        self.wait_for_audio_preload()
        self.speak_response("系统准备就绪")
        self.speak_response("小智已启动")
        # ===== 其他组件 =====
        self.motor_dialog = MotorControlDialog()   # 电机控制对话框
        self.motor_controller = MotorController()  # 电机控制器对象



    def init_ui(self):
        """初始化用户界面"""
        # 设置全局样式
        self.setStyleSheet("""
            QMainWindow {
                background-color: #0F0F0F;
                color: #E0E0E0;
            }
            QGroupBox {
                background-color: #252526;
                color: #E6E6E6;
                border: 1px solid #404040;
                border-radius: 6px;
                margin-top: 10px;
                padding-top: 15px;
                padding-bottom: 15px;
                font-weight: 500;
                font-size: 10pt;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
                color: #A0A0A0;
            }
            QLabel {
                color: #A0A0A0;
                font-size: 9pt;
            }
            QPushButton {
                background-color: #007ACC;
                color: #FFFFFF;
                border: 1px solid #005B9F;
                border-radius: 4px;
                padding: 7px 14px;
                font-weight: 500;
                font-size: 9pt;
            }
            QPushButton:hover {
                background-color: #1E90FF;
                border: 1px solid #007ACC;
            }
            QPushButton:pressed {
                background-color: #005B9F;
                border: 1px solid #004080;
            }
            QPushButton:disabled {
                background-color: #3F3F46;
                color: #707070;
                border: 1px solid #333333;
            }
            QLineEdit {
                background-color: #2D2D2D;
                border: 1px solid #404040;
                border-radius: 4px;
                padding: 6px 8px;
                color: #E6E6E6;
                font-size: 9pt;
                selection-background-color: #007ACC;
            }
            QListWidget {
                background-color: #2D2D2D;
                border: 1px solid #404040;
                border-radius: 4px;
                color: #E6E6E6;
                font-size: 9pt;
            }
            QListWidget::item {
                padding: 8px;
                border-bottom: 1px solid #404040;
            }
            QListWidget::item:selected {
                background-color: #007ACC;
                color: white;
                border: none;
            }
            QProgressBar {
                border: 1px solid #404040;
                border-radius: 4px;
                text-align: center;
                background-color: #2D2D2D;
                font-size: 9pt;
            }
            QProgressBar::chunk {
                background-color: #007ACC;
                border-radius: 4px;
            }
            QComboBox {
                background-color: #3C3C40;
                border: 1px solid #555555;
                border-radius: 4px;
                padding: 6px 10px;
                color: #E0E0E0;
                font-size: 10pt;
                selection-background-color: #007ACC;
                min-height: 30px;
            }
            QComboBox:hover {
                border: 1px solid #707070;
            }
            QComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 25px;
                border-left: 1px solid #555555;
            }
            QComboBox::down-arrow {
                image: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCA1MTIgNTEyIj48cGF0aCBmaWxsPSIjQTBBMEEwIiBkPSJNMTQ4LjggMTQ0Yy02LjQgNi40LTYuNCAxNi44IDAgMjMuMmwxMzYgMTM2YzYuNCA2LjQgMTYuOCA2LjQgMjMuMiAwbDEzNi0xMzZjNi40LTYuNCA2LjQtMTYuOCAwLTIzLjJzLTE2LjgtNi40LTIzLjIgMEwyODggMjQ3LjIgMTcyIDE0NGMtNi40LTYuNC0xNi44LTYuNC0yMy4yIDB6Ii8+PC9zdmc+);
                width: 12px;
                height: 12px;
            }
            QComboBox QAbstractItemView {
                background-color: #3C3C40;
                border: 1px solid #555555;
                color: #E0E0E0;
                selection-background-color: #007ACC;
                selection-color: white;
                outline: none;
            }
            QComboBox QAbstractItemView::item {
                padding: 8px;
                border-bottom: 1px solid #404040;
            }
            QComboBox QAbstractItemView::item:selected {
                background-color: #007ACC;
                color: white;
                border: none;
            }
            QSlider {
                padding: 0px;
                margin: 0px;
            }
            QSlider::groove:horizontal {
                background: #404040;     
                height: 6px;               
                border-radius: 3px;         
                margin: 0 8px;             
            }
            QSlider::sub-page:horizontal {
                background: #007ACC;       
                border-radius: 3px;         
            }    
            QSlider::handle:horizontal {
                background: #007ACC;     
                border: 1px solid #005B9F; 
                width: 16px;              
                height: 16px;             
                border-radius: 8px;        
                margin: -5px 0;            
            }
            QSlider::handle:horizontal:hover {
                background: #1C97EA;      
                border: 1px solid #007ACC; 
                width: 18px;                
                height: 18px;               
                margin: -6px 0;           
            }
            QSlider::handle:horizontal:pressed {
                background: #005B9F;        
                border: 1px solid #004080;  
            }  
            QSlider::groove:vertical {
                background: #404040;     
                width: 6px;               
                border-radius: 3px;       
                margin: 8px 0;            
            }
            QSlider::sub-page:vertical {
                background: #007ACC;       
                border-radius: 3px;        
            }
            QSlider::handle:vertical {
                background: #007ACC;       
                border: 1px solid #005B9F;  
                width: 16px;               
                height: 16px;             
                border-radius: 8px;        
                margin: 0 -5px;           
            }
            QSlider::handle:vertical:hover {
                background: #1C97EA;      
                border: 1px solid #007ACC;  
                width: 18px;                
                height: 18px;             
                margin: 0 -6px;             
            }
            QSlider::handle:vertical:pressed {
                background: #005B9F;       
                border: 1px solid #004080; 
            }
            QScrollArea {
                background-color: transparent;   
                border: 0px  
            }
            QScrollArea::viewport {
                background-color: #0F0F0F;
            }
            QScrollArea > QWidget > QWidget {
                background-color: #0F0F0F;
            }
            QScrollBar:vertical {
                background: transparent;       
                width: 0px;

            }

            QScrollBar::handle:vertical {
                background: transparent;       

            }

            QScrollBar::handle:vertical:hover {
                background: transparent;        
            }


            QScrollBar:horizontal {
                background: transparent;        
                height: 0px;
            }

            QScrollBar::handle:horizontal {
                background: transparent;        
            }

            QScrollBar::handle:horizontal:hover {
                background: #303030;        
            }

            QScrollBar::add-line:horizontal, 
            QScrollBar::sub-line:horizontal {
                background: none;
                width: 0px;
            }
            QSplitter::handle {
                background-color: #404040;
            }
            QCheckBox {
                color: #A0A0A0;
                font-size: 9pt;
                spacing: 5px;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
                border: 1px solid #555555;
                border-radius: 3px;
                background: #2D2D2D;
            }
            QCheckBox::indicator:checked {
                background: #007ACC;
                border: 1px solid #007ACC;
            }
            QCheckBox::indicator:unchecked:hover {
                border: 1px solid #707070;
            }
            QTabWidget::pane {
                border: 1px solid #404040;
                background: #252526;
            }
            QTabBar::tab {
                background: #2D2D2D;
                color: #A0A0A0;
                border: 1px solid #404040;
                border-bottom: none;
                padding: 5px 10px;
            }
            QTabBar::tab:selected {
                background: #252526;
                color: #E6E6E6;
                border-bottom: 2px solid #007ACC;
            }
            #cameraContainer, #centerContainer, #rightContainer {
                border: 1px solid #404040;
                border-radius: 4px;
            }
        """)


        # 主窗口设置
        self.setWindowTitle("机器人控制系统")
        self.setGeometry(100, 100, 1280, 800)
        self.setMinimumSize(1000, 700)

        # 创建中央部件和主布局
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(15, 15, 15, 15)

        # 创建主分割器
        main_splitter = QSplitter(Qt.Horizontal)

        # === 左侧区域：摄像头画面 ===
        camera_container = QWidget()
        camera_layout = QVBoxLayout(camera_container)
        camera_layout.setContentsMargins(0, 0, 0, 0)
        camera_layout.setSpacing(10)

        # 摄像头画面组
        camera_group = QGroupBox("摄像头画面")
        camera_group.setMinimumWidth(650)
        camera_group_layout = QVBoxLayout(camera_group)
        camera_group_layout.setContentsMargins(10, 15, 10, 10)

        # 摄像头标签
        self.detection_label = QLabel("摄像头未启动")
        self.detection_label.setAlignment(Qt.AlignCenter)
        self.detection_label.setMinimumSize(640, 480)
        self.detection_label.setStyleSheet("""
            background-color: #1E1E1E;
            border: 1px solid #007ACC;
            border-radius: 4px;
        """)
        camera_group_layout.addWidget(self.detection_label)

        # 摄像头控制按钮
        cam_control_layout = QHBoxLayout()
        cam_control_layout.setContentsMargins(5, 10, 5, 5)
        self.camera_status_label = QLabel("状态: 未连接")
        self.camera_status_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        self.open_camera_btn = QPushButton("启动摄像头")
        self.open_camera_btn.setFixedHeight(40)
        self.open_camera_btn.clicked.connect(self.toggle_camera)

        self.close_camera_btn = QPushButton("关闭摄像头")
        self.close_camera_btn.setFixedHeight(40)
        self.close_camera_btn.setEnabled(False)
        self.close_camera_btn.clicked.connect(self.close_camera)

        cam_control_layout.addWidget(self.camera_status_label)
        cam_control_layout.addStretch()
        cam_control_layout.addWidget(self.open_camera_btn)
        cam_control_layout.addWidget(self.close_camera_btn)

        camera_group_layout.addLayout(cam_control_layout)
        camera_layout.addWidget(camera_group)

        # === 中间区域：控制面板 ===
        control_container = QScrollArea()
        control_container.setWidgetResizable(True)
        control_container.setFrameShape(QFrame.NoFrame)

        control_content = QWidget()
        control_layout = QVBoxLayout(control_content)
        control_layout.setContentsMargins(5, 5, 5, 5)
        control_layout.setSpacing(15)

        # 示教点管理组
        teach_group = QGroupBox("示教点管理")
        teach_layout = QVBoxLayout(teach_group)
        teach_layout.setSpacing(10)

        # 保存新点区域
        save_layout = QHBoxLayout()
        self.teach_name_input = QLineEdit()
        self.teach_name_input.setPlaceholderText("输入示教点名称")
        self.save_teach_button = QPushButton("保存当前位置")
        self.save_teach_button.setFixedHeight(40)
        self.save_teach_button.setEnabled(False)
        self.save_teach_button.clicked.connect(self.save_teach_point)

        save_layout.addWidget(self.teach_name_input, 70)
        save_layout.addWidget(self.save_teach_button, 30)
        teach_layout.addLayout(save_layout)

        # 示教点列表
        self.teach_point_list = QListWidget()
        self.teach_point_list.setMinimumHeight(180)
        self.teach_point_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.teach_point_list.itemDoubleClicked.connect(self.move_to_teach_point)
        self.update_teach_point_list()
        teach_layout.addWidget(self.teach_point_list)

        # 操作按钮
        point_btns_layout = QHBoxLayout()
        self.move_button = QPushButton("移动到选定点")
        self.move_button.setFixedHeight(40)
        self.move_button.setEnabled(False)
        self.move_button.clicked.connect(self.move_to_selected_point)

        self.delete_button = QPushButton("删除选定点")
        self.delete_button.setFixedHeight(40)
        self.delete_button.clicked.connect(self.delete_selected_point)

        point_btns_layout.addWidget(self.move_button)
        point_btns_layout.addWidget(self.delete_button)
        teach_layout.addLayout(point_btns_layout)

        control_layout.addWidget(teach_group)

        # 打磨控制组
        grinding_group = QGroupBox("打磨控制")
        grinding_layout = QVBoxLayout(grinding_group)
        grinding_layout.setSpacing(10)

        # 坐标偏移设置
        offset_group = QGroupBox("坐标偏移设置")
        offset_layout = QGridLayout(offset_group)
        offset_layout.setHorizontalSpacing(10)
        offset_layout.setVerticalSpacing(8)

        offset_layout.addWidget(QLabel("X偏移(mm):"), 0, 0)
        self.offset_x_input = QLineEdit("0")
        self.offset_x_input.setPlaceholderText("X偏移")
        self.offset_x_input.setValidator(QDoubleValidator(-50, 50, 2))
        offset_layout.addWidget(self.offset_x_input, 0, 1)

        offset_layout.addWidget(QLabel("Y偏移(mm):"), 1, 0)
        self.offset_y_input = QLineEdit("0")
        self.offset_y_input.setPlaceholderText("Y偏移")
        self.offset_y_input.setValidator(QDoubleValidator(-50, 50, 2))
        offset_layout.addWidget(self.offset_y_input, 1, 1)

        offset_layout.addWidget(QLabel("Z偏移(mm):"), 2, 0)
        self.offset_z_input = QLineEdit("260")
        self.offset_z_input.setPlaceholderText("Z偏移")
        self.offset_z_input.setValidator(QDoubleValidator(-50, 50, 2))
        offset_layout.addWidget(self.offset_z_input, 2, 1)

        self.apply_offset_btn = QPushButton("应用偏移")
        self.apply_offset_btn.setFixedHeight(35)
        self.apply_offset_btn.clicked.connect(self.apply_offsets)
        offset_layout.addWidget(self.apply_offset_btn, 3, 0, 1, 2)

        grinding_layout.addWidget(offset_group)

        # 打磨参数设置
        params_group = QGroupBox("打磨参数")
        params_layout = QGridLayout(params_group)
        params_layout.setHorizontalSpacing(10)
        params_layout.setVerticalSpacing(8)

        params_layout.addWidget(QLabel("循环次数:"), 0, 0)
        self.loop_count_input = QLineEdit("1")
        self.loop_count_input.setValidator(QIntValidator(1, 100))
        params_layout.addWidget(self.loop_count_input, 0, 1)

        params_layout.addWidget(QLabel("X进深(mm):"), 1, 0)
        self.x_step_input = QLineEdit("0.0")
        self.x_step_input.setValidator(QDoubleValidator(-10.0, 10.0, 2))
        params_layout.addWidget(self.x_step_input, 1, 1)

        params_layout.addWidget(QLabel("Y进深(mm):"), 2, 0)
        self.y_step_input = QLineEdit("0.0")
        self.y_step_input.setValidator(QDoubleValidator(-10.0, 10.0, 2))
        params_layout.addWidget(self.y_step_input, 2, 1)

        params_layout.addWidget(QLabel("Z进深(mm):"), 3, 0)
        self.z_step_input = QLineEdit("0.0")
        self.z_step_input.setValidator(QDoubleValidator(-10.0, 10.0, 2))
        params_layout.addWidget(self.z_step_input, 3, 1)

        params_layout.addWidget(QLabel("路径缩放比例X:"), 4, 0)
        self.scale_factor_X_input = QLineEdit("0.6625")
        self.scale_factor_X_input.setValidator(QDoubleValidator(0.1, 10.0, 10))
        params_layout.addWidget(self.scale_factor_X_input, 4, 1)

        params_layout.addWidget(QLabel("路径缩放比例Y:"), 5, 0)
        self.scale_factor_Y_input = QLineEdit("0.56")
        self.scale_factor_Y_input.setValidator(QDoubleValidator(0.1, 10.0, 10))
        params_layout.addWidget(self.scale_factor_Y_input, 5, 1)

        distance_layout = QHBoxLayout()
        distance_layout.addWidget(QLabel("当前距离(mm):"))
        self.distance_display = QLineEdit("245")
        distance_layout.addWidget(self.distance_display)

        self.update_distance_btn = QPushButton("更新测距数据")
        self.update_distance_btn.setFixedHeight(35)
        self.update_distance_btn.clicked.connect(self.update_distance_from_sensor)
        distance_layout.addWidget(self.update_distance_btn)
        params_layout.addLayout(distance_layout, 6, 0, 1, 2)

        self.apply_grinding_params_btn = QPushButton("应用参数")
        self.apply_grinding_params_btn.setFixedHeight(35)
        self.apply_grinding_params_btn.clicked.connect(self.apply_grinding_params)
        params_layout.addWidget(self.apply_grinding_params_btn, 7, 0, 1, 2)

        grinding_layout.addWidget(params_group)

        # 历史路径管理
        history_group = QGroupBox("历史路径管理")
        history_layout = QVBoxLayout(history_group)
        history_layout.setSpacing(8)

        self.history_list = QListWidget()
        self.history_list.setMinimumHeight(120)
        history_layout.addWidget(self.history_list)

        self.use_history_checkbox = QCheckBox("使用历史路径（跳过检测）")
        self.use_history_checkbox.stateChanged.connect(self.toggle_history_path_usage)
        history_layout.addWidget(self.use_history_checkbox)

        history_btns_layout = QHBoxLayout()
        self.save_path_btn = QPushButton("保存当前路径")
        self.save_path_btn.setFixedHeight(35)
        self.save_path_btn.clicked.connect(self.save_current_path)

        self.load_path_btn = QPushButton("加载历史路径")
        self.load_path_btn.setFixedHeight(35)
        self.load_path_btn.clicked.connect(self.load_history_path)

        self.apply_path_btn = QPushButton("应用选中路径")
        self.apply_path_btn.setFixedHeight(35)
        self.apply_path_btn.clicked.connect(self.apply_history_path)

        history_btns_layout.addWidget(self.save_path_btn)
        history_btns_layout.addWidget(self.load_path_btn)
        history_btns_layout.addWidget(self.apply_path_btn)
        history_layout.addLayout(history_btns_layout)

        grinding_layout.addWidget(history_group)

        # 启动打磨按钮
        self.grinding_button = QPushButton("启动打磨")
        self.grinding_button.setFixedHeight(50)
        self.grinding_button.setFont(QFont("Arial", 12, QFont.Bold))
        self.grinding_button.clicked.connect(self.toggle_grinding)
        grinding_layout.addWidget(self.grinding_button)

        # 打磨状态指示
        status_layout = QHBoxLayout()
        status_layout.addWidget(QLabel("打磨状态:"))

        self.grinding_status_indicator = QLabel()
        self.grinding_status_indicator.setFixedSize(20, 20)
        self.grinding_status_indicator.setStyleSheet("background-color: #505050; border-radius: 10px;")

        status_layout.addWidget(self.grinding_status_indicator)
        status_layout.addStretch()

        self.grinding_progress_label = QLabel("打磨: 未运行")
        status_layout.addWidget(self.grinding_progress_label)

        grinding_layout.addLayout(status_layout)

        control_layout.addWidget(grinding_group)

        # 执行控制组
        exec_group = QGroupBox("执行控制")
        exec_layout = QVBoxLayout(exec_group)
        exec_layout.setSpacing(10)

        # XYZ移动控制
        xyz_group = QGroupBox("XYZ坐标移动")
        xyz_layout = QGridLayout(xyz_group)
        xyz_layout.setHorizontalSpacing(10)
        xyz_layout.setVerticalSpacing(8)

        xyz_layout.addWidget(QLabel("目标坐标:"), 0, 0)
        self.target_x_input = QLineEdit()
        self.target_x_input.setPlaceholderText("X")
        xyz_layout.addWidget(self.target_x_input, 0, 1)

        self.target_y_input = QLineEdit()
        self.target_y_input.setPlaceholderText("Y")
        xyz_layout.addWidget(self.target_y_input, 0, 2)

        self.target_z_input = QLineEdit()
        self.target_z_input.setPlaceholderText("Z")
        xyz_layout.addWidget(self.target_z_input, 0, 3)

        self.move_xyz_button = QPushButton("移动")
        self.move_xyz_button.setFixedHeight(35)
        self.move_xyz_button.clicked.connect(self.move_to_xyz)
        xyz_layout.addWidget(self.move_xyz_button, 0, 4)

        self.angle_correction_checkbox = QCheckBox("启用角度修正")
        self.angle_correction_checkbox.setChecked(True)
        xyz_layout.addWidget(self.angle_correction_checkbox, 1, 0, 1, 5)

        exec_layout.addWidget(xyz_group)

        # 执行选项
        options_layout = QGridLayout()
        options_layout.setHorizontalSpacing(10)
        options_layout.setVerticalSpacing(8)

        options_layout.addWidget(QLabel("运动类型:"), 0, 0)
        self.move_type_combo = QComboBox()
        self.move_type_combo.addItems(["关节运动 (MOVEJ)", "直线运动 (MOVEL)"])
        options_layout.addWidget(self.move_type_combo, 0, 1, 1, 2)

        options_layout.addWidget(QLabel("速度:"), 1, 0)
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setRange(1, 100)
        self.speed_slider.setValue(50)
        options_layout.addWidget(self.speed_slider, 1, 1)

        self.speed_label = QLabel("速度: 50")
        options_layout.addWidget(self.speed_label, 1, 2)

        exec_layout.addLayout(options_layout)

        # 执行按钮
        execute_layout = QHBoxLayout()
        self.execute_all_button = QPushButton("执行所有点")
        self.execute_all_button.setFixedHeight(40)
        self.execute_all_button.clicked.connect(self.execute_all_points)

        self.execute_selected_button = QPushButton("执行选定点")
        self.execute_selected_button.setFixedHeight(40)
        self.execute_selected_button.clicked.connect(self.execute_selected_point)

        execute_layout.addWidget(self.execute_all_button)
        execute_layout.addWidget(self.execute_selected_button)
        exec_layout.addLayout(execute_layout)

        # 控制按钮
        control_btns_layout = QHBoxLayout()
        self.pause_button = QPushButton("暂停")
        self.pause_button.setFixedHeight(40)
        self.pause_button.clicked.connect(self.pause_execution)

        self.resume_button = QPushButton("恢复")
        self.resume_button.setFixedHeight(40)
        self.resume_button.clicked.connect(self.resume_execution)

        self.stop_button = QPushButton("停止")
        self.stop_button.setFixedHeight(40)
        self.stop_button.clicked.connect(self.stop_execution)

        control_btns_layout.addWidget(self.pause_button)
        control_btns_layout.addWidget(self.resume_button)
        control_btns_layout.addWidget(self.stop_button)
        exec_layout.addLayout(control_btns_layout)

        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        exec_layout.addWidget(self.progress_bar)

        control_layout.addWidget(exec_group)

        # 设置控制内容
        control_container.setWidget(control_content)

        # === 右侧区域：连接设置 ===
        right_container = QScrollArea()
        right_container.setWidgetResizable(True)
        right_container.setFrameShape(QFrame.NoFrame)

        right_content = QWidget()
        right_layout = QVBoxLayout(right_content)
        right_layout.setContentsMargins(5, 5, 5, 5)
        right_layout.setSpacing(15)

        # 摄像头设置组
        camera_setting_group = QGroupBox("摄像头设置")
        camera_setting_layout = QVBoxLayout(camera_setting_group)
        camera_setting_layout.setSpacing(10)

        # 摄像头类型选择
        cam_type_layout = QHBoxLayout()
        cam_type_layout.addWidget(QLabel("摄像头类型:"))
        self.camera_type_combo = QComboBox()
        self.camera_type_combo.addItems(["本地摄像头", "网络摄像头"])
        self.camera_type_combo.currentIndexChanged.connect(self.change_camera_type)
        cam_type_layout.addWidget(self.camera_type_combo, 1)
        camera_setting_layout.addLayout(cam_type_layout)

        # IP地址输入
        ip_layout = QHBoxLayout()
        ip_layout.addWidget(QLabel("IP地址:"))
        self.camera_ip_input = QLineEdit("0.0.0.0")
        ip_layout.addWidget(self.camera_ip_input, 1)
        camera_setting_layout.addLayout(ip_layout)

        # 端口输入
        port_layout = QHBoxLayout()
        port_layout.addWidget(QLabel("端口:"))
        self.camera_port_input = QLineEdit("9999")
        port_layout.addWidget(self.camera_port_input, 1)
        camera_setting_layout.addLayout(port_layout)

        # 模型选择
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("ONNX模型:"))
        self.onnx_model_path_input = QLineEdit()
        self.onnx_model_path_input.setPlaceholderText("选择ONNX模型文件")
        model_layout.addWidget(self.onnx_model_path_input, 1)

        self.load_model_btn = QPushButton("加载")
        self.load_model_btn.setFixedHeight(30)
        self.load_model_btn.clicked.connect(self.load_onnx_model)
        model_layout.addWidget(self.load_model_btn)
        camera_setting_layout.addLayout(model_layout)

        # 标定按钮
        calibration_layout = QHBoxLayout()
        self.calibrate_button = QPushButton("标定")
        self.calibrate_button.setFixedHeight(40)
        self.calibrate_button.clicked.connect(self.start_calibration)

        self.single_image_calibrate_btn = QPushButton("图像标定")
        self.single_image_calibrate_btn.setFixedHeight(40)
        self.single_image_calibrate_btn.clicked.connect(self.calibrate_single_image)

        calibration_layout.addWidget(self.calibrate_button)
        calibration_layout.addWidget(self.single_image_calibrate_btn)
        camera_setting_layout.addLayout(calibration_layout)

        # 点云
        point_cloud_group = QGroupBox("点云模型")
        point_cloud_layout = QVBoxLayout(point_cloud_group)


        #取点云数据按钮
        self.get_3d_coordinates_btn = QPushButton("获取点云数据")
        self.get_3d_coordinates_btn.setFixedHeight(40)
        self.get_3d_coordinates_btn.clicked.connect(self.get_3d_coordinates)  # 连接方法
        point_cloud_layout.addWidget(self.get_3d_coordinates_btn)

        self.point_cloud_btn = QPushButton("点云配置")
        self.point_cloud_btn.setFixedHeight(40)
        self.point_cloud_btn.clicked.connect(self.open_point_cloud_config)

        point_cloud_layout.addWidget(self.point_cloud_btn)
        right_layout.addWidget(point_cloud_group)

        # 标定文件操作
        calib_file_layout = QHBoxLayout()
        self.load_calibration_recognition_btn = QPushButton("加载标定")
        self.load_calibration_recognition_btn.setFixedHeight(40)
        self.load_calibration_recognition_btn.clicked.connect(self.load_calibration_file)

        self.save_calibration_recognition_btn = QPushButton("保存标定")
        self.save_calibration_recognition_btn.setFixedHeight(40)
        self.save_calibration_recognition_btn.clicked.connect(self.save_calibration_file)

        calib_file_layout.addWidget(self.load_calibration_recognition_btn)
        calib_file_layout.addWidget(self.save_calibration_recognition_btn)
        camera_setting_layout.addLayout(calib_file_layout)

        right_layout.addWidget(camera_setting_group)

        # 机器人连接组
        connection_group = QGroupBox("机器人连接")
        connection_layout = QVBoxLayout(connection_group)
        connection_layout.setSpacing(10)

        # IP地址输入
        robot_ip_layout = QHBoxLayout()
        robot_ip_layout.addWidget(QLabel("IP地址:"))
        self.ip_input = QLineEdit(self.config.ROBOT_IP)
        self.ip_input.setPlaceholderText("192.168.25.185")
        robot_ip_layout.addWidget(self.ip_input, 1)
        connection_layout.addLayout(robot_ip_layout)

        # 端口输入
        robot_port_layout = QHBoxLayout()
        robot_port_layout.addWidget(QLabel("端口号:"))
        self.port_input = QLineEdit(str(self.config.ROBOT_PORT))
        self.port_input.setPlaceholderText("例如：8080")
        robot_port_layout.addWidget(self.port_input, 1)
        connection_layout.addLayout(robot_port_layout)

        # 连接按钮
        self.connect_button = QPushButton("连接机器人")
        self.connect_button.setFixedHeight(45)
        self.connect_button.clicked.connect(self.toggle_connection)
        connection_layout.addWidget(self.connect_button)

        # 状态指示灯
        status_layout = QHBoxLayout()
        status_layout.addWidget(QLabel("机器人状态:"))

        self.status_indicator = QLabel()
        self.status_indicator.setFixedSize(20, 20)
        self.status_indicator.setStyleSheet("background-color: #505050; border-radius: 10px;")

        status_layout.addWidget(self.status_indicator)
        status_layout.addStretch()
        connection_layout.addLayout(status_layout)

        # 控制按钮
        control_btns_layout = QGridLayout()
        control_btns_layout.setHorizontalSpacing(10)
        control_btns_layout.setVerticalSpacing(10)

        self.manual_control_btn = QPushButton("手动控制")
        self.manual_control_btn.setFixedHeight(40)
        self.manual_control_btn.clicked.connect(self.open_manual_control)

        self.motor_control_btn = QPushButton("电机控制")
        self.motor_control_btn.setFixedHeight(40)
        self.motor_control_btn.clicked.connect(self.open_motor_control)

        control_btns_layout.addWidget(self.manual_control_btn, 0, 0)
        control_btns_layout.addWidget(self.motor_control_btn, 0, 1)

        connection_layout.addLayout(control_btns_layout)

        right_layout.addWidget(connection_group)

        # 语音控制组
        recognition_group = QGroupBox("语音控制")
        recognition_layout = QVBoxLayout(recognition_group)
        recognition_layout.setSpacing(10)

        self.speech_recognition_btn = QPushButton("启动语音识别")
        self.speech_recognition_btn.setFixedHeight(40)
        self.speech_recognition_btn.clicked.connect(self.toggle_speech_recognition)

        self.calibration_recognition_btn = QPushButton("校准麦克风")
        self.calibration_recognition_btn.setFixedHeight(40)
        self.calibration_recognition_btn.clicked.connect(self.calibrate_microphone)

        self.load_calibration_recognition_btn = QPushButton("加载麦克风校准配置")
        self.load_calibration_recognition_btn.setFixedHeight(40)
        self.load_calibration_recognition_btn.clicked.connect(self.load_calibration_settings)

        recognition_layout.addWidget(self.speech_recognition_btn)
        recognition_layout.addWidget(self.calibration_recognition_btn)
        recognition_layout.addWidget(self.load_calibration_recognition_btn)

        right_layout.addWidget(recognition_group)

        # 设置右侧内容
        right_container.setWidget(right_content)

        # 检测按钮
        detect_group = QGroupBox("目标检测")
        detect_layout = QVBoxLayout(detect_group)

        self.detect_button = QPushButton("启动检测")
        self.detect_button.setFixedHeight(40)
        self.detect_button.clicked.connect(self.toggle_detection)

        self.detect_image_btn = QPushButton("检测图像")
        self.detect_image_btn.setFixedHeight(40)
        self.detect_image_btn.clicked.connect(self.detect_image)

        detect_layout.addWidget(self.detect_button)
        detect_layout.addWidget(self.detect_image_btn)

        right_layout.addWidget(detect_group)

        # === 组装主界面 ===
        main_splitter.addWidget(camera_container)
        main_splitter.addWidget(control_container)
        main_splitter.addWidget(right_container)

        # 设置分割器初始大小
        main_splitter.setSizes([650, 500, 300])

        main_layout.addWidget(main_splitter)

        # 状态栏
        self.status_bar = self.statusBar()
        self.status_label = QLabel("系统就绪")
        self.status_bar.addPermanentWidget(self.status_label)

        self.audio_status_label = QLabel("麦克风: 未连接")
        self.status_bar.addPermanentWidget(self.audio_status_label)

        # 连接信号
        self.speed_slider.valueChanged.connect(self.update_speed_label)



    # ========== TCP 服务端（大项目用来接收信号） ==========
    def start_stop_server(self,host="0.0.0.0", port=50007):
        """
        启动TCP服务端，接收小项目的 STOP 信号
        :param host: 绑定的IP地址，默认监听所有网卡
        :param port: 监听的端口号，需与小项目一致
        """
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
            server_socket.bind((host, port))
            server_socket.listen(1)
            print(f"🟢 STOP 服务端已启动，等待小项目连接... (端口 {port})")

            while True:
                conn, addr = server_socket.accept()
                with conn:
                    print(f"📡 来自 {addr} 的连接")
                    data = conn.recv(1024)
                    if not data:
                        continue
                    message = data.decode("utf-8").strip()
                    if message == "STOP":
                        print("🛑 收到 STOP 信号！急停机器人！")
                        # TODO: 在这里写你的处理代码

    def wait_for_audio_preload(self):
        """等待音频预加载完成"""
        while not self.audio_system.preload_complete:
            time.sleep(0.1)
            QApplication.processEvents()  # 保持UI响应


    def calibrate_microphone(self):
        """启动麦克风校准"""
        # 如果已有校准在进行，则先停止
        if self.calibrator and self.calibrator.isRunning():
            self.calibrator.stop_calibration = True
            self.calibrator.wait(1000)

        # 创建新的校准器
        self.calibrator = MicrophoneCalibrator()
        self.calibrator.status_updated.connect(self.update_calibration_status)
        self.calibrator.calibration_done.connect(self.handle_calibration_result)
        self.calibrator.start()
        self.status_label.setText("麦克风校准已启动...")
        self.speak_response("背景噪音校准中，请保持安静")

    def update_calibration_status(self, message):
        """更新校准状态"""
        self.status_label.setText(message)
        print(f"校准状态: {message}")

    def handle_calibration_result(self, background, speech, threshold):
        """处理校准结果"""
        self.background_level = background
        self.speech_level = speech
        self.audio_threshold = threshold

        self.status_label.setText(f"校准完成! 阈值={threshold:.4f}")
        print(f"校准结果: 背景噪音={background:.4f}, 说话音量={speech:.4f}, 阈值={threshold:.4f}")
        self.speak_response("校准完成，阈值已设定")
        # 保存校准结果到配置文件
        self.save_calibration_settings()

    def save_calibration_settings(self):
        """保存校准设置到配置文件"""
        settings = {
            'background_level': self.background_level,
            'speech_level': self.speech_level,
            'threshold': self.audio_threshold,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
        }

        try:
            with open('microphone_calibration.json', 'w') as f:
                json.dump(settings, f, indent=4)
            print("校准设置已保存")
        except Exception as e:
            print(f"保存校准设置失败: {str(e)}")

    def load_calibration_settings(self):
        """从配置文件加载校准设置"""
        try:
            if os.path.exists('microphone_calibration.json'):
                with open('microphone_calibration.json', 'r') as f:
                    settings = json.load(f)

                self.background_level = settings.get('background_level', 0.01)
                self.speech_level = settings.get('speech_level', 0.05)
                self.audio_threshold = settings.get('threshold', 0.02)

                print(f"加载校准设置: 阈值={self.audio_threshold:.4f}")
                return True
        except Exception as e:
            print(f"加载校准设置失败: {str(e)}")

        return False

    def stop_recording_for_processing(self):
        """停止录音以便处理"""
        if self.recording_thread:
            self.recording_thread.stop_recording = True
        self.recording_timer.stop()

    def update_scaling_factors(self, distance):
        """根据测距距离更新缩放比例"""
        try:
            # 确保距离有效（大于10mm）
            if distance < 10:
                distance = self.base_distance

            self.current_distance = distance

            # 计算新缩放比例（保留4位小数）
            new_scale_x = round((self.base_distance * self.base_scale_x) / distance, 4)
            new_scale_y = round((self.base_distance * self.base_scale_y) / distance, 4)

            # 更新UI显示
            self.scale_factor_X_input.setText(str(new_scale_x))
            self.scale_factor_Y_input.setText(str(new_scale_y))

            # 更新内存变量
            self.path_scale_factor_X = new_scale_x
            self.path_scale_factor_Y = new_scale_y

            print(f"距离{distance}mm处缩放比例更新: X={new_scale_x}, Y={new_scale_y}")
            return True
        except Exception as e:
            print(f"更新缩放比例失败: {str(e)}")
            return False

    def update_distance_from_sensor(self):
        """从测距传感器获取最新距离并更新比例"""
        try:
            distance = float(self.distance_display.text())

            try:
                # 初始化传感器
                tof = TOF_Sense("/dev/ttyUSB0", 921600)

                # 使用类方法获取距离
                distance = tof.get_distance()
                if distance is not None:
                    print(f"距离: {distance} mm")
                else:
                    print("未能获取有效距离数据")

            except KeyboardInterrupt:
                print("程序已退出")

            # 更新UI显示
            self.distance_display.setText(str(distance))

            # 自动更新缩放比例
            self.update_scaling_factors(distance)
            return True

        except Exception as e:
            QMessageBox.warning(self, "测距失败", f"无法获取距离数据: {str(e)}")
            return False


    def open_point_cloud_config(self):
        """打开点云配置对话框"""

    def toggle_history_path_usage(self, state):
        """切换历史路径使用状态"""
        self.use_history_path = (state == Qt.Checked)
        if self.use_history_path and not self.current_history_path:
            QMessageBox.warning(self, "未选择路径", "请先选择并应用一个历史路径")
            self.use_history_checkbox.setChecked(False)
            self.use_history_path = False
        else:
            status = "启用" if self.use_history_path else "禁用"
            self.status_label.setText(f"历史路径使用状态: {status}")

    def save_current_path(self):
        """保存当前路径到历史记录"""
        if not hasattr(self.detection_thread, 'detection_system') or not self.detection_thread.detection_system.fixed_contour:
            QMessageBox.warning(self, "无路径", "当前没有可保存的路径")
            return

        # 获取路径名称
        name, ok = QInputDialog.getText(self, "路径命名", "输入路径名称:")
        if not ok or not name:
            return

        # 获取当前时间
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

        # 获取当前旋转角度
        rotation = self.coordinate_rotation

        # 创建路径数据对象
        path_data = {
            "name": name,
            "timestamp": timestamp,
            "points": self.detection_thread.detection_system.fixed_contour,
            "original_size": self.detection_thread.detection_system.fixed_contour_size,
            "rotation": rotation,  # 保存旋转角度
            "world_coords": []  # 存储转换后的世界坐标
        }

        # 添加到历史列表
        self.history_paths.append(path_data)
        self.update_history_list()

        # 保存到文件
        try:
            with open("history_paths.json", "w") as f:
                json.dump(self.history_paths, f)
            QMessageBox.information(self, "保存成功", f"已保存路径: {name}")
        except Exception as e:
            QMessageBox.warning(self, "保存失败", f"无法保存历史路径: {str(e)}")

    def load_history_path(self):
        """从文件加载历史路径"""
        try:
            with open("history_paths.json", "r") as f:
                self.history_paths = json.load(f)
            self.update_history_list()
            QMessageBox.information(self, "加载成功", f"已加载 {len(self.history_paths)} 条历史路径")
        except FileNotFoundError:
            QMessageBox.warning(self, "文件未找到", "历史路径文件不存在")
        except Exception as e:
            QMessageBox.warning(self, "加载失败", f"无法加载历史路径: {str(e)}")

    def update_history_list(self):
        """更新历史路径列表显示"""
        self.history_list.clear()
        for path in self.history_paths:
            points_count = len(path["points"])
            item = QListWidgetItem(f"{path['name']} - {path['timestamp']} ({points_count}点)")
            item.setData(Qt.UserRole, path)
            self.history_list.addItem(item)

    def apply_history_path(self):
        """应用选中的历史路径"""
        selected = self.history_list.currentItem()
        if not selected:
            QMessageBox.warning(self, "未选择", "请先选择一条历史路径")
            return

        path_data = selected.data(Qt.UserRole)
        self.current_history_path = path_data

        # 更新使用历史路径复选框状态
        self.use_history_checkbox.setChecked(True)
        self.use_history_path = True

        QMessageBox.information(self, "路径应用", f"已应用路径: {path_data['name']}")

    def apply_grinding_params(self, show_message=True):
        """应用打磨参数（show_message控制是否显示提示框）"""
        try:
            # 获取循环次数
            loops = int(self.loop_count_input.text())
            if loops < 1 or loops > 100:
                if show_message:
                    QMessageBox.warning(self, "参数错误", "循环次数必须在1-100之间")
                return False
            self.grinding_loops = loops

            # 获取X进深
            x_step = float(self.x_step_input.text())
            if abs(x_step) > 10:
                if show_message:
                    QMessageBox.warning(self, "参数错误", "X进深不能超过±10mm")
                return False
            self.grinding_x_step = x_step

            # 获取Y进深
            y_step = float(self.y_step_input.text())
            if abs(y_step) > 10:
                if show_message:
                    QMessageBox.warning(self, "参数错误", "Y进深不能超过±10mm")
                return False
            self.grinding_y_step = y_step

            # 获取Z进深
            z_step = float(self.z_step_input.text())
            if abs(z_step) > 10:
                if show_message:
                    QMessageBox.warning(self, "参数错误", "Z进深不能超过±10mm")
                return False
            self.grinding_z_step = z_step

            # 获取缩放比例
            scale_factor_X = float(self.scale_factor_X_input.text())
            scale_factor_Y = float(self.scale_factor_Y_input.text())
            if scale_factor_X < 0.1 or scale_factor_X > 10.0:
                if show_message:
                    QMessageBox.warning(self, "参数错误", "缩放比例必须在0.1-10.0之间")
                return False
            self.path_scale_factor_X = scale_factor_X
            if scale_factor_Y < 0.1 or scale_factor_Y > 10.0:
                if show_message:
                    QMessageBox.warning(self, "参数错误", "缩放比例必须在0.1-10.0之间")
                return False
            self.path_scale_factor_Y = scale_factor_Y

            # 只在需要时显示消息框
            if show_message:
                QMessageBox.information(self, "参数设置",
                                    f"打磨参数已更新:\n循环次数: {self.grinding_loops}\n"
                                    f"X进深: {self.grinding_x_step}mm\n"
                                    f"Y进深: {self.grinding_y_step}mm\n"
                                    f"Z进深: {self.grinding_z_step}mm\n"
                                    f"X缩放比例: {scale_factor_X}\n"
                                    f"Y缩放比例: {scale_factor_Y}")

            return True
        except ValueError:
            if show_message:
                QMessageBox.warning(self, "输入错误", "请输入有效的数字参数")
            return False

    def update_frame(self, frame):
        """更新检测画面"""
        # 将OpenCV图像转换为Qt图像
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        q_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)

        # 缩放图像以适应标签大小
        scaled_pixmap = pixmap.scaled(
            self.detection_label.width(),
            self.detection_label.height(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )

        self.detection_label.setPixmap(scaled_pixmap)

    def start_calibration(self):
        """启动摄像头标定"""
        # 确保检测线程已经启动
        if not self.detection_thread or not self.detection_thread.isRunning():
            # 先创建检测线程
            self.create_detection_thread()

            # 如果线程创建成功，启动它
            if self.detection_thread:
                self.detection_thread.start()
                self.detect_button.setText("停止检测")
                self.detect_button.setStyleSheet("background-color: #FF4D4D;")

        # 确保检测线程已正确初始化
        if self.detection_thread and self.detection_thread.isRunning():
            try:
                self.detection_thread.perform_calibration()
            except Exception as e:
                QMessageBox.critical(self, "标定错误", f"标定过程中发生错误: {str(e)}")
        else:
            QMessageBox.warning(self, "错误", "无法启动标定，请先确保摄像头检测已正常启动")

    def update_detection_result(self, result):
        """更新检测结果文本"""

    # 添加新的标定文件操作方法
    def load_calibration_file(self):
        """加载标定文件"""
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择标定文件", "",
            "JSON文件 (*.json);;所有文件 (*)",
            options=options
        )

        if file_path:
            if self.detection_thread and self.detection_thread.detection_system:
                if self.detection_thread.detection_system.load_calibration_params(file_path):
                    QMessageBox.information(self, "加载成功", "标定参数已加载")
                    # 更新UI状态
                    self.calibration_status.setText("标定状态: 已加载")
                else:
                    QMessageBox.warning(self, "加载失败", "无法加载标定参数")

    def save_calibration_file(self):
        """保存标定文件"""
        options = QFileDialog.Options()
        file_path = QFileDialog.getSaveFileName(
            self, "保存标定文件", "",
            "JSON文件 (*.json);;所有文件 (*)",
            options=options
        )

        if file_path:
            if self.detection_thread and self.detection_thread.detection_system:
                if self.detection_thread.detection_system.save_calibration_params(file_path):
                    QMessageBox.information(self, "保存成功", f"标定参数已保存到:\n{file_path}")
                else:
                    QMessageBox.warning(self, "保存失败", "无法保存标定参数")

    def toggle_detection(self):
        if hasattr(self, 'camera_thread') and self.camera_thread and self.camera_thread.isRunning():
            self.close_camera()
            time.sleep(0.5)
        if self.detect_button.text() == "启动检测":
            try:
                # 获取摄像头类型和参数
                camera_type = "network" if self.camera_type_combo.currentIndex() == 1 else "local"
                ip = self.camera_ip_input.text()
                port = int(self.camera_port_input.text())

                # 创建检测线程，传入正确的参数
                self.detection_thread = ONNXDetectionThread(
                    config=self.config,
                    model_path=self.onnx_model_path,
                    ip=ip,
                    port=port
                )
                self.detection_thread.camera_type = camera_type
                # 连接信号
                self.detection_thread.update_frame.connect(self.update_frame)
                self.detection_thread.detection_result.connect(self.update_detection_result)
                self.detection_thread.detection_coords.connect(self.handle_detection_coords)
                # 确保线程被正确创建
                if not self.detection_thread:
                    raise RuntimeError("无法创建检测线程")
                # 启动线程
                self.detection_thread.start()
                self.detect_button.setText("停止检测")
                self.detect_button.setStyleSheet("background-color: #FF4D4D;")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"无法启动目标检测: {str(e)}")
                print(self, "错误", f"无法启动目标检测: {str(e)}",flush=True)
                self.detection_thread.stop()
                self.detection_thread.wait(2000)  # 等待线程安全退出
                self.detection_thread = None
                self.detect_button.setText("启动检测")
                self.detect_button.setStyleSheet("")
                self.detection_label.clear()
                self.detection_label.setText("摄像头未启动")
                self.detection_result.setText("检测已停止")
                cv2.destroyAllWindows()
        else:
            # 停止摄像头检测
            if self.detection_thread and self.detection_thread.isRunning():
                self.detection_thread.stop()
                self.detection_thread.wait(2000)  # 等待线程安全退出
                self.detection_thread = None
                self.detect_button.setText("启动检测")
                self.detect_button.setStyleSheet("")
                self.detection_label.clear()
                self.detection_label.setText("摄像头未启动")
                self.detection_label.setText("检测已停止")
                cv2.destroyAllWindows()


    def handle_single_image_result(self, result_frame, result_text):
        """处理单张图像检测结果"""
        if result_frame is None:
            QMessageBox.warning(self, "错误", result_text)
            self.detection_label.setText("检测失败")
            self.detection_result.setText(result_text)
            return

        # 显示结果图像
        self.display_image(result_frame)

        # 显示检测结果
        self.detection_result.setText("检测结果: " + result_text)

    def display_image(self, frame):
        """显示图像在QLabel中"""
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        q_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)

        # 缩放图像以适应标签大小
        scaled_pixmap = pixmap.scaled(
            self.detection_label.width(),
            self.detection_label.height(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.detection_label.setPixmap(scaled_pixmap)

    def load_yolo_model(self):
        """加载YOLO模型"""
        options = QFileDialog.Options()
        model_path, _ = QFileDialog.getOpenFileName(
            self, "选择YOLO模型文件", "",
            "模型文件 (*.pt);;所有文件 (*)",
            options=options
        )

        if model_path:
            global YOLO_MODEL_PATH
            YOLO_MODEL_PATH = model_path
            QMessageBox.information(self, "成功", f"已加载模型: {model_path}")

            # 重新初始化检测线程
            if self.detection_thread:
                self.detection_thread.stop()
                self.detection_thread = None
                self.detect_button.setText("启动检测")
                self.detect_button.setStyleSheet("")

    def camera_settings(self):
        """摄像头设置"""
        QMessageBox.information(self, "摄像头设置", "当前使用默认摄像头")

    def update_speed_label(self, value):
        """更新速度标签"""
        self.speed_label.setText(f"速度: {value}")

    def update_ui_state(self, connected):
        """更新UI状态"""
        if connected:
            self.connect_button.setText("断开连接")
            self.grinding_button.setEnabled(True)
            self.save_teach_button.setEnabled(True)
            self.move_button.setEnabled(True)
            self.execute_all_button.setEnabled(True)
            self.execute_selected_button.setEnabled(True)
            self.status_indicator.setStyleSheet("background-color: #00FF00; border-radius: 10px;")
        else:
            self.connect_button.setText("链接机器人")
            self.grinding_button.setEnabled(False)
            self.save_teach_button.setEnabled(False)
            self.move_button.setEnabled(False)
            self.execute_all_button.setEnabled(False)
            self.execute_selected_button.setEnabled(False)
            self.status_indicator.setStyleSheet("background-color: #FF0000; border-radius: 10px;")

    def toggle_connection(self):
        """切换连接状态"""
        if self.connect_button.text() == "链接机器人":
            ip = self.ip_input.text() or self.config.ROBOT_IP
            port = int(self.port_input.text() or self.config.ROBOT_PORT)
            if not ip or not port:
                QMessageBox.warning(self, "输入错误", "请填写IP地址和端口号！")
                return

            print(f"正在连接到 {ip}:{port}...")
            thread_server = threading.Thread(target=self.start_stop_server)
            thread_server.start()
            try:
                self.connection = RobotConnection(ip, port)
                if self.connection.is_connected():
                    self.update_ui_state(True)
                else:
                    QMessageBox.critical(self, "连接失败", "无法连接到机械臂，请检查IP和端口")
                    self.speak_response("连接失败，请检查网络和电源")
            except Exception as e:
                QMessageBox.critical(self, "连接错误", f"连接过程中发生错误: {str(e)}")
        else:
            print("断开机器人连接...")
            if self.connection:
                self.connection.disconnect()
                self.connection.stop_heartbeat()
            self.update_ui_state(False)
            self.stop_speech_recognition()

    def open_manual_control(self):
        """打开手动控制对话框"""
        if not self.connection or not self.connection.is_connected():
            self.speak_response("机器人未连接，请先建立连接")
            QMessageBox.warning(self, "未连接", "请先连接机械臂")
            return

        dialog = ManualControlDialog(self.connection, self)
        dialog.exec()

    def open_motor_control(self):
        """打开电机控制对话框"""
        self.motor_dialog.show()

    def load_teach_points(self):
        """从文件加载保存的示教点"""
        try:
            if os.path.exists(self.config.TEACH_POINTS_FILE):
                with open(self.config.TEACH_POINTS_FILE, 'r') as f:
                    self.teach_points = json.load(f)
                print(f"成功加载 {len(self.teach_points)} 个示教点")
            else:
                self.teach_points = []
                print("没有找到示教点文件，将创建新文件")
        except Exception as e:
            print(f"加载示教点失败: {str(e)}")
            self.teach_points = []

    def save_teach_points(self):
        """保存示教点到文件"""
        try:
            with open(self.config.TEACH_POINTS_FILE, 'w') as f:
                json.dump(self.teach_points, f, indent=4)
            print("示教点已保存到文件")
            return True
        except Exception as e:
            print(f"保存示教点失败: {str(e)}")
            return False

    def update_teach_point_list(self):
        """更新示教点列表显示"""
        self.teach_point_list.clear()
        for point in self.teach_points:
            name = point.get('name', '未命名')
            if 'positions' in point:
                positions = ", ".join([f"{p:.2f}" for p in point['positions']])
            elif 'angles' in point:
                positions = ", ".join([f"{p:.2f}" for p in point['angles']])
            else:
                positions = "未知位置"
            time_str = point.get('timestamp', point.get('time', '未知时间'))
            item = QListWidgetItem(f"{name} - {time_str}\n位置: [{positions}]")
            self.teach_point_list.addItem(item)

    def save_teach_point_object(self, point):
        """保存示教点对象"""
        if any(p['name'] == point['name'] for p in self.teach_points):
            QMessageBox.warning(self, "名称重复", f"示教点名称 '{point['name']}' 已存在，请使用不同的名称")
            return

        self.teach_points.append(point)
        if self.save_teach_points():
            self.update_teach_point_list()
            QMessageBox.information(self, "保存成功", f"成功保存示教点: {point['name']}")
        else:
            self.teach_points.pop()
            QMessageBox.critical(self, "保存失败", "无法保存示教点到文件")

    def save_teach_point(self):
        """保存当前位置为示教点"""
        if not self.connection or not self.connection.is_connected():
            QMessageBox.warning(self, "未连接", "未连接到机器人，无法保存示教点")
            return

        name = self.teach_name_input.text().strip()
        if not name:
            QMessageBox.warning(self, "输入错误", "请输入示教点名称")
            return

        if any(p.get('name') == name for p in self.teach_points):
            QMessageBox.warning(self, "名称重复", f"示教点名称 '{name}' 已存在，请使用不同的名称")
            return

        try:
            mc = self.connection.get_robot()
            coords = mc.get_coords()
            angles = mc.get_angles()

            if angles:
                point = {
                    'name': name,
                    'coords': coords,
                    'angles': angles,
                    'positions': angles,
                    'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
                }
                self.teach_points.append(point)

                if self.save_teach_points():
                    self.update_teach_point_list()
                    QMessageBox.information(self, "保存成功", f"成功保存示教点: {name}")
                    self.teach_name_input.clear()
                else:
                    self.teach_points.pop()
                    QMessageBox.critical(self, "保存失败", "无法保存示教点到文件")
            else:
                QMessageBox.warning(self, "获取位置失败", "无法获取机器人当前位置")
        except Exception as e:
            QMessageBox.critical(self, "保存错误", f"保存示教点时出错: {str(e)}")

    def get_selected_point(self):
        """获取选定的示教点"""
        selected_items = self.teach_point_list.selectedItems()
        if not selected_items:
            return None
        index = self.teach_point_list.row(selected_items[0])
        if 0 <= index < len(self.teach_points):
            return self.teach_points[index]
        return None

    def move_to_selected_point(self):
        """移动到选定的示教点"""
        point = self.get_selected_point()
        if point:
            self.move_to_teach_point(point)

    def move_to_teach_point(self, point=None):
        """移动到指定的示教点"""
        if not self.connection or not self.connection.is_connected():
            QMessageBox.warning(self, "未连接", "未连接到机器人，无法移动")
            return

        if point is None:
            selected_items = self.teach_point_list.selectedItems()
            if selected_items:
                idx = self.teach_point_list.row(selected_items[0])
                point = self.teach_points[idx] if idx < len(self.teach_points) else None

        if not point:
            return

        try:
            point_name = point.get('name', '未命名点位')
            print(f"移动到: {point_name}")
            mc = self.connection.get_robot()
            target = point.get('positions', point.get('angles', []))

            # 添加关节角度限制检查
            safe_target = self.apply_joint_limits(target)

            # 使用安全角度
            mc.send_angles(safe_target, 50)

            # 高亮显示
            for i in range(self.teach_point_list.count()):
                if self.teach_points[i].get('name') == point.get('name'):
                    item = self.teach_point_list.item(i)
                    item.setBackground(QBrush(QColor("#007ACC")))
                    break

            QMessageBox.information(self, "移动成功", f"正在移动到: {point_name}")
        except Exception as e:
            QMessageBox.critical(self, "移动失败", f"移动过程中出错: {str(e)}")

    def apply_joint_limits(self, angles):
        """应用关节角度限制确保安全移动"""
        # 关节角度限制
        limits = [
            (-168, 168),  # 关节1
            (-135, 135),  # 关节2
            (-150, 150),  # 关节3
            (-145, 145),  # 关节4
            (-165, 165),  # 关节5
            (-180, 180)  # 关节6
        ]

        safe_angles = []
        for i, angle in enumerate(angles):
            if i < len(limits):
                min_val, max_val = limits[i]
                # 确保角度在限制范围内
                safe_angle = max(min_val, min(angle, max_val))
                safe_angles.append(safe_angle)
            else:
                safe_angles.append(angle)

        return safe_angles

    def delete_selected_point(self):
        """删除选定的示教点"""
        point = self.get_selected_point()
        if point:
            reply = QMessageBox.question(self, "确认删除",
                                         f"确定要删除示教点 '{point['name']}' 吗?",
                                         QMessageBox.Yes | QMessageBox.No)
            if reply == QMessageBox.Yes:
                self.teach_points = [p for p in self.teach_points if p['name'] != point['name']]
                if self.save_teach_points():
                    self.update_teach_point_list()
                    QMessageBox.information(self, "删除成功", f"已删除示教点: {point['name']}")
                else:
                    self.load_teach_points()
                    QMessageBox.critical(self, "删除失败", "无法保存更改")

    def toggle_grinding(self):
        """切换打磨状态"""
        # 在启动打磨前检查连接对象
        if not self.connection or not hasattr(self.connection, 'reconnect'):
            QMessageBox.warning(self, "连接错误", "机器人连接对象无效")
            return

        current_text = self.grinding_button.text()
        if current_text == "启动打磨":
            print("启动打磨程序...")
            self.grinding_button.setText("停止打磨")
            self.grinding_button.setStyleSheet("background-color: #FF4D4D;")
            self.grinding_status_indicator.setStyleSheet("background-color: #00FF00; border-radius: 10px;")

            # 获取三维坐标数据
            camera_coordinates_3d = self.get_3d_coordinates()

            if not camera_coordinates_3d:
                QMessageBox.warning(self, "坐标错误", "无法获取有效的三维坐标数据")
                # 恢复按钮状态
                self.grinding_button.setText("启动打磨")
                self.grinding_button.setStyleSheet("")
                self.grinding_status_indicator.setStyleSheet("background-color: #FF0000; border-radius: 10px;")
                return

            print(f"获取到 {len(camera_coordinates_3d)} 个三维坐标点")

            # 初始化打磨控制器
            if not hasattr(self, 'grinding_controller') or self.grinding_controller is None:
                robot_instance = self.connection.get_robot()
                self.grinding_controller = MyCobotGrindingController(robot_instance)

            # 使用线程池提交打磨任务
            task_id = self.thread_pool.submit_task(
                self._run_grinding_procedure,
                camera_coordinates_3d,
                self.user_offset_x,
                self.user_offset_y,
                self.user_offset_z
            )

            # 存储当前任务ID用于后续管理
            self.current_grinding_task_id = task_id
            self.task_ids['grinding'] = task_id
            self.speak_response("打磨程序已启动")

            self.DAMCX()

        else:
            print("停止打磨程序...")
            self._stop_grinding_procedure()

    def DAMCX(self):
        T = np.array([
            [-2.74280449e-02, -9.99601611e-01, -6.65752354e-03, 6.48263423e+01],
            [-9.99259510e-01, 2.72376602e-02, 2.71761376e-02, 2.04309745e+02],
            [-2.69839756e-02, 7.39798203e-03, -9.99608491e-01, 8.88548188e+02],
            [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

        # 直接使用矩阵求逆
        def transform_point_inverse(T, point):
            """使用逆矩阵变换点"""
            # 将点转换为齐次坐标
            point_homogeneous = np.append(point, 1.0)

            # 计算变换矩阵的逆
            T_inv = np.linalg.inv(T)

            # 应用逆变换
            robot_point_homogeneous = T_inv @ point_homogeneous

            # 转换回3D坐标 (齐次坐标除以w分量)
            robot_point = robot_point_homogeneous[:3] / robot_point_homogeneous[3]

            return robot_point

        try:
            mc = self.connection.get_robot()
            # 获取当前位置
            # current_pos = controller.get_current_position()
            # print(f"当前位置: {current_pos}")

            mc.power_on()

            import time

            def move_to_point_clouds(point_cloud_list, speed=20, fixed_height=None):
                """
                将点云坐标列表转换为机器人坐标并使用直线运动模式依次运动到每个位置

                Args:
                    point_cloud_list: 点云坐标列表，每个元素为[x, y, z]格式的数组
                    speed: 直线运动速度（1-100）
                    fixed_height: 固定高度，如果为None则使用点云中的Z坐标
                """

                # 坐标补偿函数 - 基于x和y中绝对值较大的值判断
                def get_compensation_value(x, y):
                    max_abs_value = max(abs(x), abs(y))
                    if max_abs_value <= 100:
                        return 0  # 100mm以内不需要补偿
                    elif max_abs_value <= 200:
                        return 0  # 100-200mm补偿2mm
                    else:
                        return 0  # 200mm以上补偿4mm

                # 坐标范围限制函数
                def limit_coordinate(value, min_val, max_val):
                    return max(min(value, max_val), min_val)

                # 计算两点之间的距离
                def distance_between_points(p1, p2):
                    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2) ** 0.5

                # 存储所有笛卡尔坐标点
                cartesian_points = []

                print(f"开始处理 {len(point_cloud_list)} 个点...")

                # 计算所有点的笛卡尔坐标
                for i, point_cloud in enumerate(point_cloud_list):
                    print(f"计算第 {i + 1}/{len(point_cloud_list)} 个点的坐标...")

                    # 将点云坐标转换为机器人坐标系
                    robot_coords_inverse = transform_point_inverse(T, point_cloud)
                    # print("原机器人坐标:" , robot_coords_inverse)
                    # 获取补偿值（x和y使用相同的补偿值）
                    compensation_value = get_compensation_value(robot_coords_inverse[0], robot_coords_inverse[1])
                    z_compensation = 0  # z补偿暂时设为0

                    # 应用补偿
                    compensated_x = robot_coords_inverse[0] + compensation_value
                    compensated_y = robot_coords_inverse[1] + compensation_value
                    compensated_z = robot_coords_inverse[2] + z_compensation

                    # 限制坐标范围
                    x_limited = limit_coordinate(compensated_x, -281.45, 281.45)
                    y_limited = limit_coordinate(compensated_y, -281.45, 281.45)
                    z_limited = limit_coordinate(compensated_z, -70, 412.67)

                    # 计算最终坐标
                    coords = [x_limited - 8, y_limited, z_limited + 130 + 5, 0, 180, 0]

                    # 只有在明确需要固定高度时才使用固定高度
                    if fixed_height is not None:
                        coords[2] = fixed_height

                    cartesian_points.append(coords)

                    print(f"点 {i + 1}: 世界坐标 {point_cloud} -> 机器人坐标 {coords}")

                # # 检查点之间的距离，过滤过于接近的点
                # filtered_points = [cartesian_points[0]]
                # for i in range(1, len(cartesian_points)):
                #     dist = distance_between_points(cartesian_points[i], filtered_points[-1])
                #     if dist > 5:  # 只保留距离大于5mm的点
                #         filtered_points.append(cartesian_points[i])
                #     else:
                #         print(f"跳过点 {i + 1}，与上一个点距离太近: {dist:.2f}mm")

                filtered_points = cartesian_points.copy()
                print(f"过滤后剩余 {len(filtered_points)} 个点")

                if len(filtered_points) == 0:
                    print("没有有效的点需要运动")
                    return

                # 使用直线运动模式执行所有点
                print(f"\n开始直线轨迹运动，共 {len(filtered_points)} 个点...")

                # 先移动到第一个点（使用非线性模式确保可达）
                print(f"准备运动到起始点...")
                mc.sync_send_angles([0.17, 31.2, -116.19, -13.27, 1.66, 116.63], 30, timeout=1)
                mc.sync_send_angles([39.55, -0.17, -59.76, -34.27, 2.54, 128.05], 30, timeout=1)
                result = mc.sync_send_coords(filtered_points[0], speed, mode=0, timeout=0.5)
                if result != 1:
                    print("移动到起始点失败！")
                    return

                time.sleep(0.5)

                # 然后使用直线运动模式依次运动到其他点
                success_count = 0
                for i, coords in enumerate(filtered_points):
                    if i == 0:  # 第一个点已经处理过
                        success_count += 1
                        continue

                    print(f"直线运动到第 {i + 1}/{len(filtered_points)} 个点...")

                    # 使用直线运动模式（mode=1）
                    result = mc.sync_send_coords(coords, speed, mode=1, timeout=0.05)

                    if result == 1:
                        success_count += 1
                        print(f"成功到达点 {i + 1}")
                    else:
                        print(f"运动到点 {i + 1} 失败，尝试使用非线性模式...")
                        # 如果直线运动失败，尝试非线性模式
                        result = mc.sync_send_coords(coords, speed, mode=0, timeout=0.05)
                        if result == 1:
                            success_count += 1
                            print(f"使用非线性模式成功到达点 {i + 1}")
                        else:
                            print(f"点 {i + 1} 完全运动失败，跳过该点")

                    # 在点之间添加短暂停顿（可选）
                    if i < len(filtered_points) - 1:
                        time.sleep(0.1)

                print(f"\n运动完成！成功到达 {success_count}/{len(filtered_points)} 个点")
                mc.sync_send_angles([39.55, -0.17, -59.76, -34.27, 2.54, 128.05], 30, timeout=1)
                mc.sync_send_angles([0.17, 31.2, -116.19, -13.27, 1.66, 116.63], 30, timeout=1)

            height = 790  # 固定高度值
            # 定义点云坐标列表
            point_clouds = [
                # np.array([-85.107887, 78.239586, 870.622925]),
                # np.array([-2.596224, 20.766823, 868.374023]),
                # np.array([-40.959755, -13.706027, 868.326233]),
                # np.array([-115.630562, 5.762152, 869.585449]),
                # np.array([250,236.941483, 895.233643]),
                # np.array([250,236.941483, 895.233643]),
                # np.array([12.415163,-60.981064, 868.282288]),
                # np.array([135.874268,-37.065334, 868.502136]),
                # np.array([103.7227, 23.619373, 750.0898]),
                # np.array([103.722702, 23.61937332, 793.515564]),
                # np.array([45.334656, -40.731045, 790.805969]),

                # 周五用
                np.array([19.08568304, 4.909702972, height]),
                np.array([16.92098316, 4.270843018, height]),
                np.array([15.20927383, 2.799748592, height]),
                np.array([14.25223887, 0.755695824, height]),
                np.array([14.21855292, -1.501056908, height]),
                np.array([15.11415302, -3.572763471, height]),
                np.array([16.78119226, -5.094291586, height]),
                np.array([18.9258598, -5.79747633, height]),
                np.array([21.1701642, -5.558383429, height]),
                np.array([23.1185533, -4.419152314, height]),
                np.array([24.42762918, -2.580569169, height]),
                np.array([24.866671, -0.366678947, height]),
                np.array([24.35829891, 1.832326614, height]),
                np.array([22.99211203, 3.628879154, height]),
                np.array([21.00889682, 4.706341492, height]),
                np.array([18.75818924, 4.874814012, height]),
                np.array([16.63666996, 4.104603922, height]),
                np.array([15.01825061, 2.531458517, height]),
                np.array([14.18817293, 0.432640119, height]),
                np.array([14.29273568, -1.821940611, height]),
                np.array([15.31351, -3.834920349, height]),
                np.array([17.07058737, -5.251517248, height]),
                np.array([19.25428798, -5.822060209, height]),
                np.array([21.47974093, -5.445992687, height]),
                np.array([23.35471659, -4.189595495, height]),
                np.array([24.54875593, -2.274304996, height]),
                np.array([24.85141297, -0.037685594, height]),
                np.array([24.20934529, 2.126065022, height]),
                np.array([22.73571545, 3.835592073, height]),
                np.array([20.69024645, 4.789596369, height]),
                np.array([18.43344627, 4.819937417, height]),
                np.array([16.36306941, 3.921267694, height]),
                np.array([14.84401378, 2.251975137, height]),
                np.array([14.14400855, 0.106267722, height]),
                np.array([14.38642761, -2.137679835, height]),
                np.array([15.52854529, -4.084378276, height]),
                np.array([17.36906668, -5.390727642, height]),
                np.array([19.5836052, -5.826487641, height]),
                np.array([21.78185486, -5.314856835, height]),
                np.array([23.57638051, -3.946008672, height]),
                np.array([24.65090223, -1.961198675, height]),
                np.array([24.81603865, 0.289756143, height]),
                np.array([24.04268498, 2.410131518, height]),
                np.array([22.46714255, 4.026217431, height]),
                np.array([20.36709615, 4.853183422, height]),
                np.array([18.11267287, 4.74527913, height]),
                np.array([16.1012083, 3.721522375, height]),
                np.array([14.68721722, 1.962347319, height]),
                np.array([14.11991148, -0.22219653, height]),
                np.array([14.49927707, -2.447089649, height]),
                np.array([15.75845189, -4.320201069, height]),
                np.array([17.67551004, -5.511400329, height]),
                np.array([19.91257558, -5.810742011, height]),
                np.array([22.07537217, -5.16546801, height]),
                np.array([23.78271319, -3.689305999, height]),
                np.array([24.73368472, -1.642425257, height]),
                np.array([24.7606808, 0.614417415, height]),
                np.array([23.85894343, 2.683460034, height]),
                np.array([22.18740123, 4.200039833, height]),
                np.array([20.04065865, 4.896864015, height]),
                np.array([17.79707286, 4.651119336, height]),
                np.array([15.85206936, 3.506117585, height]),
                np.array([14.54844937, 1.663662003, height]),
                np.array([14.11597215, -0.551519951, height]),
                np.array([14.63086056, -2.749008875, height]),
                np.array([16.00236699, -4.541503711, height]),
                np.array([17.98876742, -5.613082438, height]),
                np.array([20.23996452, -5.77488241, height]),
                np.array([22.35919133, -4.998386849, height]),
                np.array([23.97294027, -3.420450849, height]),
                np.array([24.79679274, -1.31918106, height]),
                np.array([24.68554716, 0.935079807, height]),
                np.array([23.65881021, 2.945024803, height]),
                np.array([21.89754133, 4.356406945, height]),
                np.array([19.71215904, 4.92047422, height]),
                np.array([17.48783065, 4.537811405, height]),
                np.array([15.61658758, 3.275861712, height]),
                np.array([14.42823102, 1.357040118, height]),
                np.array([14.13220535, -0.880466629, height]),
                np.array([14.78068426, -3.042304446, height]),
                np.array([16.2593752, -4.747455681, height]),
                np.array([18.30766318, -5.695392369, height]),
                np.array([20.56454339, -5.719043414, height]),
                np.array([22.63224722, -4.814240389, height]),
                np.array([24.14634786, -3.140452204, height]),
                np.array([24.83998946, -0.99267918, height]),
                np.array([24.5909197, 1.250539911, height]),
                np.array([23.4430364, 3.193844202, height]),
                np.array([21.59865067, 4.494731941, height]),
                np.array([19.38283014, 4.923925433, height]),
                np.array([17.18610678, 4.405780568, height]),
                np.array([15.39564671, 3.031618878, height]),
                np.array([14.32701332, 1.043632379, height]),
                np.array([14.16855016, -1.207802066, height]),
                np.array([14.9481859, -3.32587566, height]),
                np.array([16.528512, -4.937284067, height]),
                np.array([18.63100056, -5.758021223, height]),
                np.array([20.88509407, -5.643434582, height]),
                np.array([22.89351507, -4.613719708, height]),
                np.array([24.30228518, -2.850360865, height]),
                np.array([24.86311275, -0.664144939, height]),
                np.array([24.47715354, 1.559613844, height]),
                np.array([23.21243176, 3.428984442, height]),
                np.array([21.29185093, 4.614495702, height]),
                np.array([19.05390789, 4.907204699, height]),
                np.array([16.8930336, 4.25552232, height]),
                np.array([15.19007589, 2.774305697, height]),
                np.array([14.24517613, 0.724614968, height]),
                np.array([14.22487017, -1.532297814, height]),
                np.array([15.13273687, -3.598658308, height]),
                np.array([16.80876735, -5.110276463, height]),
                np.array([18.95756611, -5.80073396, height]),
                np.array([21.20041357, -5.548339663, height]),
                np.array([23.14201438, -4.397577338, height]),
                np.array([24.44016702, -2.551265509, height]),
                np.array([24.86607584, -0.334811288, height]),
                np.array([24.34467565, 1.861141688, height]),
                np.array([22.96786172, 3.649563071, height]),
                np.array([20.97829352, 4.715248771, height]),
                np.array([18.72662668, 4.870374771, height]),

            ]

            # 执行运动
            move_to_point_clouds(point_clouds)
        except Exception as e:
            print(f"发生错误: {e}")

    def _stop_grinding_procedure(self):
        """停止打磨程序"""
        # 更新UI状态
        self.grinding_button.setText("启动打磨")
        self.grinding_button.setStyleSheet("")
        self.grinding_status_indicator.setStyleSheet("background-color: #FF0000; border-radius: 10px;")

        # 停止机械臂
        if hasattr(self, 'grinding_controller') and self.grinding_controller:
            try:
                self.grinding_controller.mc.stop()
            except Exception as e:
                print(f"停止机械臂时出错: {e}")

        # 停止电机
        if hasattr(self, 'motor_controller') and self.motor_controller:
            try:
                self.motor_controller.emergency_stop()
            except Exception as e:
                print(f"停止电机时出错: {e}")

    def _run_grinding_procedure(self, camera_coordinates_3d, user_offset_x, user_offset_y, user_offset_z=0):
        """使用线程池的打磨函数 - 使用点云直接运动模式"""
        if not self.connection or not self.connection.is_connected():
            print("未连接到机器人，无法执行打磨")
            return

        # 配置标定矩阵
        self.T = np.array([
            [-2.74280449e-02, -9.99601611e-01, -6.65752354e-03, 6.48263423e+01],
            [-9.99259510e-01, 2.72376602e-02, 2.71761376e-02, 2.04309745e+02],
            [-2.69839756e-02, 7.39798203e-03, -9.99608491e-01, 8.88548188e+02],
            [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
        ])

        def transform_point_inverse(T, point):
            """使用逆矩阵变换点"""
            point_homogeneous = np.append(point, 1.0)
            T_inv = np.linalg.inv(T)
            robot_point_homogeneous = T_inv @ point_homogeneous
            robot_point = robot_point_homogeneous[:3] / robot_point_homogeneous[3]
            return robot_point

        # 坐标补偿函数 - 基于x和y中绝对值较大的值判断
        def get_compensation_value(x, y):
            max_abs_value = max(abs(x), abs(y))
            if max_abs_value <= 100:
                return 0  # 100mm以内不需要补偿
            elif max_abs_value <= 200:
                return 0  # 100-200mm补偿2mm
            else:
                return 0  # 200mm以上补偿4mm

        # 坐标范围限制函数
        def limit_coordinate(value, min_val, max_val):
            return max(min(value, max_val), min_val)

        # 计算两点之间的距离
        def distance_between_points(p1, p2):
            return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2) ** 0.5

        # 电机控制标志
        motor_started = False

        # 获取机器人控制对象
        try:
            mc = self.connection.get_robot()  # 获取实际的机器人控制对象
            if mc is None:
                print("无法获取机器人控制对象")
                return
        except Exception as e:
            print(f"获取机器人控制对象失败: {str(e)}")
            return

        try:
            # 启动电机
            if hasattr(self, 'motor_controller') and self.motor_controller:
                try:
                    motor_speed = getattr(self, 'motor_max_speed', 400)
                    print(f"启动打磨电机，转速: {motor_speed} RPM")
                    self.motor_controller.stop()
                    self.motor_controller.set_speed(motor_speed)
                    self.motor_controller.forward()
                    motor_started = True
                    time.sleep(0.5)
                except Exception as motor_err:
                    print(f"启动电机失败: {motor_err}")

            # 检查坐标数据
            if not camera_coordinates_3d or len(camera_coordinates_3d) == 0:
                print("没有提供有效的三维坐标数据")
                return

            print(f"开始处理 {len(camera_coordinates_3d)} 个三维坐标点")
            print(f"使用的用户偏移量: X={user_offset_x}, Y={user_offset_y}, Z={user_offset_z}")

            # 存储所有笛卡尔坐标点
            cartesian_points = []

            # 计算所有点的笛卡尔坐标
            for i, point_cloud in enumerate(camera_coordinates_3d):
                # 检查停止请求
                if hasattr(self, 'current_grinding_task_id'):
                    task_status = self.thread_pool.get_task_status(self.current_grinding_task_id)
                    if task_status != 'pending':  # 如果任务不再是pending状态，说明被取消了
                        print("收到停止请求，终止坐标计算")
                        return

                print(f"计算第 {i + 1}/{len(camera_coordinates_3d)} 个点的坐标...")

                # 将点云坐标转换为机器人坐标系
                try:
                    robot_coords_inverse = transform_point_inverse(self.T, point_cloud)
                    print(f"点云坐标 {point_cloud} -> 机器人坐标 {robot_coords_inverse}")
                except Exception as e:
                    print(f"坐标转换失败: {str(e)}")
                    continue

                # 应用用户偏移量
                center_x = robot_coords_inverse[0] + user_offset_x
                center_y = robot_coords_inverse[1] + user_offset_y
                center_z = robot_coords_inverse[2] + user_offset_z

                # 获取补偿值
                compensation_value = get_compensation_value(center_x, center_y)

                # 应用补偿
                compensated_x = center_x + compensation_value
                compensated_y = center_y + compensation_value
                compensated_z = center_z

                # 限制坐标范围
                x_limited = limit_coordinate(compensated_x, -281.45, 281.45)
                y_limited = limit_coordinate(compensated_y, -281.45, 281.45)
                z_limited = limit_coordinate(compensated_z, -70, 412.67)

                # 计算最终坐标
                coords = [x_limited, y_limited, z_limited, 0, 180, 0]
                cartesian_points.append(coords)

                print(f"点 {i + 1}: 最终坐标 {coords}")

            filtered_points = cartesian_points.copy()
            print(f"过滤后剩余 {len(filtered_points)} 个点")

            if len(filtered_points) == 0:
                print("没有有效的点需要运动")
                return

            # 使用直线运动模式执行所有点
            print(f"\n开始直线轨迹运动，共 {len(filtered_points)} 个点...")

            # 先移动到安全位置
            print("移动到安全位置...")
            mc.sync_send_angles([0.17, 31.2, -116.19, -13.27, 1.66, 116.63], 30, timeout=2)
            mc.sync_send_angles([39.55, -0.17, -59.76, -34.27, 2.54, 128.05], 30, timeout=2)

            # 移动到第一个点的安全高度
            if len(filtered_points) > 0:
                first_point_safe = filtered_points[0].copy()
                first_point_safe[2] += 100  # Z轴高度+100
                print(f"移动到第一个点的安全高度: {first_point_safe}")
                result = mc.sync_send_coords(first_point_safe, 20, mode=0, timeout=2)
                if result != 1:
                    print("移动到安全高度失败，终止运动")
                    return

            # 依次运动到所有点
            success_count = 0
            for i, coords in enumerate(filtered_points):
                # 检查停止请求
                if hasattr(self, 'current_grinding_task_id'):
                    task_status = self.thread_pool.get_task_status(self.current_grinding_task_id)
                    if task_status != 'pending':
                        print("收到停止请求，终止运动")
                        return

                print(f"直线运动到第 {i + 1}/{len(filtered_points)} 个点...")

                try:
                    # 使用同步直线运动模式
                    result = mc.sync_send_coords(coords, 20, mode=0, timeout=2)
                    time.sleep(0.1)  # 短暂停顿，确保运动完成

                    if result == 1:
                        success_count += 1
                        print(f"成功到达点 {i + 1}")
                    else:
                        print(f"点 {i + 1} 运动失败，跳过该点")
                except Exception as coord_err:
                    print(f"直线运动失败: {coord_err}，跳过该点")

            print(f"\n运动完成！成功到达 {success_count}/{len(filtered_points)} 个点")

            # 返回到安全位置
            print("返回到安全位置...")

            # 先抬升到安全高度
            if len(filtered_points) > 0:
                last_point_safe = filtered_points[-1].copy()
                last_point_safe[2] += 100
                mc.sync_send_coords(last_point_safe, 20, mode=0, timeout=2)

            # 经过过渡点返回安全位置
            mc.sync_send_angles([39.55, -0.17, -59.76, -34.27, 2.54, 128.05], 30, timeout=2)
            mc.sync_send_angles([0.17, 31.2, -116.19, -13.27, 1.66, 116.63], 30, timeout=2)

            safe_position = [110.4, -56.0, 205.6, 179.53, -6.68, 151.89]
            result = mc.sync_send_coords(safe_position, 50, 0, 2)

            print("所有坐标点打磨完成")
            return "success"  # 返回成功结果

        except Exception as e:
            print(f"打磨过程中发生错误: {str(e)}")
            import traceback
            traceback.print_exc()
            raise e
        finally:
            # 停止电机
            if motor_started and hasattr(self, 'motor_controller') and self.motor_controller:
                try:
                    print("停止打磨电机...")
                    self.motor_controller.emergency_stop()
                except Exception as motor_err:
                    print(f"停止电机时出错: {motor_err}")
            print("打磨线程完成")

    def get_3d_coordinates(self, server_host='localhost', server_port=8888):
        """获取三维坐标数据的方法"""
        # try:
        #     detector = PointCloudDefectDetector(
        #         defect_api_url="http://192.168.25.184:9000",
        #         camera_api_url="http://192.168.25.184:8000",
        #         standard_part_path=r"E:\pointcloud_ai_project2\data\npy\standard_part.npy"
        #     )
        #
        #     # 执行完整流程
        #     result = detector.get_3d_coordinates()
        #
        #
        #     if result['success']:
        #         print(f"✓ 操作成功!")
        #         print(f"扫描ID: {result['scan_id']}")
        #         print(f"服务端路径: {result['server_path']}")
        #         print(f"点云文件: {result['pointcloud_file']}")
        #         print(f"发现缺陷点数量: {result['num_defects']}")
        #         print(f"单位: {result['unit']}")
        #
        #         # 打印所有缺陷点
        #         if result['defect_points']:
        #             print(f"\n所有 {len(result['defect_points'])} 个缺陷点坐标:")
        #             for i, point in enumerate(result['defect_points']):
        #                 print(f"缺陷点 {i + 1}: ({point['x']:.3f}, {point['y']:.3f}, {point['z']:.3f}), "
        #                       f"距离: {point['distance']:.3f}{result['unit']}")
        #     else:
        #         print(f"✗ 操作失败: {result['message']}")
        #         if 'server_path' in result:
        #             print(f"服务端路径: {result.get('server_path', '未知')}")
        #         if 'pointcloud_file' in result:
        #             print(f"点云文件: {result.get('pointcloud_file', '未知')}")
        #         if 'tried_files' in result:
        #             print(f"尝试的文件: {result.get('tried_files', [])}")
        #
        #     return result
        # except Exception as e:
        #     print(f"获取3D坐标时发生错误: {e}")
        #     return {'success': False, 'message': str(e)}

        DY = [
    [19.08568304, 4.909702972, 790],
    [16.92098316, 4.270843018, 790],
    [15.20927383, 2.799748592, 790],
    [14.25223887, 0.755695824, 790],
    [14.21855292, -1.501056908, 790],
    [15.11415302, -3.572763471, 790],
    [16.78119226, -5.094291586, 790],
    [18.9258598, -5.79747633, 790],
    [21.1701642, -5.558383429, 790],
    [23.1185533, -4.419152314, 790],
    [24.42762918, -2.580569169, 790],
    [24.866671, -0.366678947, 790],
    [24.35829891, 1.832326614, 790],
    [22.99211203, 3.628879154, 790],
    [21.00889682, 4.706341492, 790],
    [18.75818924, 4.874814012, 790],
    [16.63666996, 4.104603922, 790],
    [15.01825061, 2.531458517, 790],
    [14.18817293, 0.432640119, 790],
    [14.29273568, -1.821940611, 790],
    [15.31351, -3.834920349, 790],
    [17.07058737, -5.251517248, 790],
    [19.25428798, -5.822060209, 790],
    [21.47974093, -5.445992687, 790],
    [23.35471659, -4.189595495, 790],
    [24.54875593, -2.274304996, 790],
    [24.85141297, -0.037685594, 790],
    [24.20934529, 2.126065022, 790],
    [22.73571545, 3.835592073, 790],
    [20.69024645, 4.789596369, 790],
    [18.43344627, 4.819937417, 790],
    [16.36306941, 3.921267694, 790],
    [14.84401378, 2.251975137, 790],
    [14.14400855, 0.106267722, 790],
    [14.38642761, -2.137679835, 790],
    [15.52854529, -4.084378276, 790],
    [17.36906668, -5.390727642, 790],
    [19.5836052, -5.826487641, 790],
    [21.78185486, -5.314856835, 790],
    [23.57638051, -3.946008672, 790],
    [24.65090223, -1.961198675, 790],
    [24.81603865, 0.289756143, 790],
    [24.04268498, 2.410131518, 790],
    [22.46714255, 4.026217431, 790],
    [20.36709615, 4.853183422, 790],
    [18.11267287, 4.74527913, 790],
    [16.1012083, 3.721522375, 790],
    [14.68721722, 1.962347319, 790],
    [14.11991148, -0.22219653, 790],
    [14.49927707, -2.447089649, 790],
    [15.75845189, -4.320201069, 790],
    [17.67551004, -5.511400329, 790],
    [19.91257558, -5.810742011, 790],
    [22.07537217, -5.16546801, 790],
    [23.78271319, -3.689305999, 790],
    [24.73368472, -1.642425257, 790],
    [24.7606808, 0.614417415, 790],
    [23.85894343, 2.683460034, 790],
    [22.18740123, 4.200039833, 790],
    [20.04065865, 4.896864015, 790],
    [17.79707286, 4.651119336, 790],
    [15.85206936, 3.506117585, 790],
    [14.54844937, 1.663662003, 790],
    [14.11597215, -0.551519951, 790],
    [14.63086056, -2.749008875, 790],
    [16.00236699, -4.541503711, 790],
    [17.98876742, -5.613082438, 790],
    [20.23996452, -5.77488241, 790],
    [22.35919133, -4.998386849, 790],
    [23.97294027, -3.420450849, 790],
    [24.79679274, -1.31918106, 790],
    [24.68554716, 0.935079807, 790],
    [23.65881021, 2.945024803, 790],
    [21.89754133, 4.356406945, 790],
    [19.71215904, 4.92047422, 790],
    [17.48783065, 4.537811405, 790],
    [15.61658758, 3.275861712, 790],
    [14.42823102, 1.357040118, 790],
    [14.13220535, -0.880466629, 790],
    [14.78068426, -3.042304446, 790],
    [16.2593752, -4.747455681, 790],
    [18.30766318, -5.695392369, 790],
    [20.56454339, -5.719043414, 790],
    [22.63224722, -4.814240389, 790],
    [24.14634786, -3.140452204, 790],
    [24.83998946, -0.99267918, 790],
    [24.5909197, 1.250539911, 790],
    [23.4430364, 3.193844202, 790],
    [21.59865067, 4.494731941, 790],
    [19.38283014, 4.923925433, 790],
    [17.18610678, 4.405780568, 790],
    [15.39564671, 3.031618878, 790],
    [14.32701332, 1.043632379, 790],
    [14.16855016, -1.207802066, 790],
    [14.9481859, -3.32587566, 790],
    [16.528512, -4.937284067, 790],
    [18.63100056, -5.758021223, 790],
    [20.88509407, -5.643434582, 790],
    [22.89351507, -4.613719708, 790],
    [24.30228518, -2.850360865, 790],
    [24.86311275, -0.664144939, 790],
    [24.47715354, 1.559613844, 790],
    [23.21243176, 3.428984442, 790],
    [21.29185093, 4.614495702, 790],
    [19.05390789, 4.907204699, 790],
    [16.8930336, 4.25552232, 790],
    [15.19007589, 2.774305697, 790],
    [14.24517613, 0.724614968, 790],
    [14.22487017, -1.532297814, 790],
    [15.13273687, -3.598658308, 790],
    [16.80876735, -5.110276463, 790],
    [18.95756611, -5.80073396, 790],
    [21.20041357, -5.548339663, 790],
    [23.14201438, -4.397577338, 790],
    [24.44016702, -2.551265509, 790],
    [24.86607584, -0.334811288, 790],
    [24.34467565, 1.861141688, 790],
    [22.96786172, 3.649563071, 790],
    [20.97829352, 4.715248771, 790],
    [18.72662668, 4.870374771, 790]
]
        return DY

    def _point_cloud_worker(self):
        """点云数据获取的工作线程函数"""
        return self.get_3d_coordinates()

    def keh(self):
        """获取点云数据的主方法 - 使用线程池"""
        # 检查是否已有任务在运行
        if self.task_ids.get('point_cloud') is not None:
            print("点云数据获取任务已在运行")
            return

        # 显示加载提示
        self.show_loading_message("正在获取3D坐标数据...")
        time.sleep(3)
        # 提交任务到线程池
        task_id = self.thread_pool.submit_task(self._point_cloud_worker)
        self.task_ids['point_cloud'] = task_id
        print(f"启动点云数据获取任务，ID: {task_id}")

    def _handle_point_cloud_completed(self, result):
        """处理点云数据获取任务完成"""
        try:
            # 确保关闭加载提示
            self.hide_loading_message()

            # 调用display方法
            self.display()

            # 显示结果弹窗
            self.show_result_dialog(result)

            print("点云数据获取任务完成，结果:", result)
        except Exception as e:
            print(f"处理点云数据结果时出错: {e}")
            self.hide_loading_message()

    def _handle_point_cloud_failed(self, exception):
        """处理点云数据获取任务失败"""
        try:
            self.hide_loading_message()
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.warning(self, "错误", f"获取3D坐标时发生错误: {str(exception)}")
        except Exception as e:
            print(f"处理点云数据获取失败时出错: {e}")

    def _process_3d_data_result(self, dy):
        """处理3D数据结果（在主线程中执行）"""
        try:
            # 确保关闭加载提示
            self.hide_loading_message()

            # 调用display方法
            self.display()

            # 显示结果弹窗
            self.show_result_dialog(dy)

            print("获取到的数据:", dy)
        except Exception as e:
            print(f"处理3D数据结果时出错: {e}")
            self.hide_loading_message()

    def _handle_3d_data_error(self, error_msg):
        """处理3D数据错误（在主线程中执行）"""
        try:
            self.hide_loading_message()
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.warning(self, "错误", f"获取3D坐标时发生错误: {error_msg}")
        except Exception as e:
            print(f"处理错误时发生异常: {e}")

    def show_loading_message(self, message="处理中..."):
        """显示加载提示 - 直接使用主界面样式表"""
        try:
            # 先关闭可能存在的旧对话框
            self.hide_loading_message()

            from PyQt5.QtWidgets import QMessageBox, QApplication, QLabel, QVBoxLayout, QDialog, QProgressBar
            from PyQt5.QtCore import Qt

            # 创建自定义加载对话框
            self.loading_dialog = QDialog(self)
            self.loading_dialog.setWindowTitle("请稍候")
            self.loading_dialog.setModal(True)
            self.loading_dialog.setFixedSize(300, 120)
            self.loading_dialog.setWindowFlags(Qt.Dialog | Qt.CustomizeWindowHint | Qt.WindowTitleHint)

            # 直接应用主界面的样式表
            self.loading_dialog.setStyleSheet(self.styleSheet())

            # 创建布局和控件
            layout = QVBoxLayout(self.loading_dialog)

            # 添加消息标签
            label = QLabel(message)
            label.setAlignment(Qt.AlignCenter)
            label.setStyleSheet("font-size: 10pt; padding: 10px;")
            layout.addWidget(label)

            # 添加进度条（不确定模式）
            progress_bar = QProgressBar()
            progress_bar.setRange(0, 0)  # 不确定模式
            progress_bar.setTextVisible(False)
            layout.addWidget(progress_bar)

            # 显示对话框
            self.loading_dialog.show()

            # 强制处理事件，确保对话框显示
            QApplication.processEvents()
        except Exception as e:
            print(f"显示加载消息时出错: {e}")

    def hide_loading_message(self):
        """隐藏加载提示 - 安全版本"""
        try:
            if hasattr(self, 'loading_dialog') and self.loading_dialog:
                self.loading_dialog.close()
                self.loading_dialog.deleteLater()
                del self.loading_dialog
        except Exception as e:
            print(f"隐藏加载消息时出错: {e}")

    def show_result_dialog(self, result):
        """显示结果弹窗 - 直接使用主界面样式表"""
        try:
            from PyQt5.QtWidgets import (QTextEdit, QVBoxLayout, QDialog,
                                         QPushButton, QScrollArea, QWidget, QLabel, QHBoxLayout)
            from PyQt5.QtCore import Qt

            # 创建自定义对话框
            dialog = QDialog(self)
            dialog.setWindowTitle("3D坐标获取结果")
            dialog.setMinimumWidth(700)
            dialog.setMinimumHeight(600)

            # 直接应用主界面的样式表
            dialog.setStyleSheet(self.styleSheet())

            # 主布局
            main_layout = QVBoxLayout(dialog)

            # 标题
            title_label = QLabel("3D坐标获取结果")
            title_label.setStyleSheet("""
                QLabel {
                    font-size: 14pt;
                    font-weight: bold;
                    padding: 10px;
                    background-color: #252526;
                    border-radius: 3px;
                }
            """)
            title_label.setAlignment(Qt.AlignCenter)
            main_layout.addWidget(title_label)

            # 基本信息区域
            info_widget = QWidget()
            info_layout = QVBoxLayout(info_widget)

            if result['success']:
                status_text = "✓ 3D坐标获取成功！"
                status_style = "color: #4EC9B0; font-weight: bold;"
            else:
                status_text = "✗ 3D坐标获取失败"
                status_style = "color: #F44747; font-weight: bold;"

            status_label = QLabel(status_text)
            status_label.setStyleSheet(f"QLabel {{ {status_style} font-size: 11pt; padding: 5px; }}")
            info_layout.addWidget(status_label)

            # 详细信息
            if result['success']:
                details_text = f"""
                <table>
                <tr><td style="color: #9CDCFE; padding-right: 10px;">扫描ID:</td><td>{result['scan_id']}</td></tr>
                <tr><td style="color: #9CDCFE; padding-right: 10px;">缺陷点数量:</td><td>{result['num_defects']}</td></tr>
                <tr><td style="color: #9CDCFE; padding-right: 10px;">单位:</td><td>{result['unit']}</td></tr>
                </table>
                """
            else:
                details_text = f"错误信息: {result.get('message', '未知错误')}"

            details_label = QLabel(details_text)
            details_label.setStyleSheet("QLabel { font-size: 10pt; padding: 5px; }")
            info_layout.addWidget(details_label)

            main_layout.addWidget(info_widget)

            # 添加所有缺陷点（如果成功且有数据）
            if result['success'] and result['defect_points']:
                # 添加分隔线
                separator = QLabel()
                separator.setFrameStyle(QLabel.HLine)
                separator.setStyleSheet("QLabel { background-color: #404040; }")
                separator.setFixedHeight(1)
                main_layout.addWidget(separator)

                # 缺陷点标题
                defects_title = QLabel(f"所有缺陷点坐标 ({len(result['defect_points'])} 个)")
                defects_title.setStyleSheet(
                    "QLabel { font-size: 11pt; font-weight: bold; color: #CE9178; padding: 5px; }")
                main_layout.addWidget(defects_title)

                # 创建滚动区域显示所有缺陷点
                scroll_area = QScrollArea()
                scroll_widget = QWidget()
                scroll_layout = QVBoxLayout(scroll_widget)

                # 添加所有缺陷点
                defects_text = ""
                for i, point in enumerate(result['defect_points']):
                    defects_text += f"缺陷点 {i + 1}:\n"
                    defects_text += f"  坐标: ({point['x']:.3f}, {point['y']:.3f}, {point['z']:.3f})\n"
                    defects_text += f"  距离: {point['distance']:.3f}{result['unit']}\n"
                    defects_text += "-" * 60 + "\n\n"

                defects_edit = QTextEdit()
                defects_edit.setPlainText(defects_text)
                defects_edit.setReadOnly(True)
                defects_edit.setStyleSheet("font-family: Consolas, 'Courier New', monospace; font-size: 9pt;")
                scroll_layout.addWidget(defects_edit)

                scroll_area.setWidget(scroll_widget)
                scroll_area.setWidgetResizable(True)
                main_layout.addWidget(scroll_area)

            # 添加确定按钮
            button_layout = QHBoxLayout()
            button_layout.addStretch()

            ok_button = QPushButton("确定")
            ok_button.clicked.connect(dialog.accept)
            button_layout.addWidget(ok_button)

            button_layout.addStretch()
            main_layout.addLayout(button_layout)

            dialog.exec_()

        except Exception as e:
            print(f"显示结果对话框时出错: {e}")
            # 回退到简单消息框，使用主界面样式
            from PyQt5.QtWidgets import QMessageBox
            msg_box = QMessageBox(self)
            msg_box.setWindowTitle("3D坐标获取结果")
            msg_box.setStyleSheet(self.styleSheet())

            if result['success']:
                msg_box.setIcon(QMessageBox.Information)
                msg_box.setText("操作成功完成！")
            else:
                msg_box.setIcon(QMessageBox.Warning)
                msg_box.setText(f"操作失败: {result.get('message', '未知错误')}")

            msg_box.exec_()

    def display(self):

        try:
            image_path = "/home/er/MasterComputer/UI/default (3).jfif"
            if os.path.exists(image_path):
                pixmap = QPixmap(image_path)

                # 缩放图像以适应标签大小
                scaled_pixmap = pixmap.scaled(
                    self.detection_label.width(),
                    self.detection_label.height(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )

                self.detection_label.setPixmap(scaled_pixmap)
                self.detection_label.setText("")
            else:
                print(f"图片点云不存在: {image_path}")

        except Exception as e:
            print(f"显示点云失败: {str(e)}")


    def process_history_path_3d(self):
        """从历史路径处理三维坐标"""
        if not self.current_history_path:
            return []

        # 假设历史路径中存储的是三维坐标
        if 'world_coords' in self.current_history_path and self.current_history_path['world_coords']:
            return self.current_history_path['world_coords']

        # 如果历史路径中只有二维点，转换为三维
        elif 'points' in self.current_history_path and self.current_history_path['points']:
            points_3d = []
            for point in self.current_history_path['points']:
                if len(point) >= 2:
                    # 将二维点转换为三维点，Z坐标设为0
                    points_3d.append([point[0], point[1], 0])
            return points_3d

        return []

    def contour_to_3d(self, contour_points):
        """将轮廓点转换为三维坐标"""
        points_3d = []
        for point in contour_points:
            if len(point) >= 2:
                # 将二维轮廓点转换为三维点，Z坐标设为0
                points_3d.append([point[0], point[1], 0])
        return points_3d

    def ensure_detection_system(self):
        """确保检测系统可用"""
        # 如果检测线程和检测系统已经存在，直接返回
        if hasattr(self, 'detection_thread') and hasattr(self.detection_thread, 'detection_system'):
            return True

        # 创建并初始化检测系统
        try:
            # 创建检测线程
            self.detection_thread = ONNXDetectionThread(self.onnx_model_path)

            # 加载模型
            if not self.detection_thread.load_model():
                QMessageBox.warning(self, "模型加载失败", "无法加载ONNX模型")
                return False

            # 加载标定参数
            self.detection_thread.detection_system.load_calibration_params()

            print("检测系统已初始化（用于历史路径处理）")
            return True
        except Exception as e:
            QMessageBox.critical(self, "初始化失败", f"无法初始化检测系统: {str(e)}")
            return False

    def process_history_path(self):
        """处理历史路径用于打磨"""
        if not self.current_history_path:
            QMessageBox.warning(self, "未选择路径", "请先选择并应用一个历史路径")
            return None

        print(f"处理历史路径: {self.current_history_path['name']}")

        # 确保检测系统可用
        if not self.ensure_detection_system():
            QMessageBox.critical(self, "检测系统错误", "无法初始化检测系统")
            return None

        # 检查路径点
        if not self.current_history_path.get('points'):
            QMessageBox.warning(self, "路径数据错误", "历史路径点为空")
            return None

        # 检查原始尺寸
        if not self.current_history_path.get('original_size'):
            QMessageBox.warning(self, "路径数据错误", "历史路径缺少原始尺寸信息")
            return None

        rotation = self.current_history_path.get('rotation', 90)  # 默认为90度

        # 尝试重新计算世界坐标
        print("重新计算历史路径的世界坐标...")
        world_coords = self.convert_path_to_world_coords(
            self.current_history_path['points'],
            self.current_history_path['original_size']
        )

        if world_coords is None:
            QMessageBox.critical(self, "转换失败", "无法将历史路径点转换为世界坐标。请检查标定参数。")
            return None

        # 更新历史路径的世界坐标
        self.current_history_path['world_coords'] = world_coords

        # 更新存储的历史路径
        self.save_history_paths()

        print(f"成功处理历史路径，生成 {len(world_coords)} 个世界坐标点")
        return world_coords

    def convert_path_to_world_coords(self, pixel_points, original_size):
        """使用 CameraDetectionSystem 的转换方法"""
        if not hasattr(self.detection_thread, 'detection_system'):
            print("错误：检测系统不可用")
            return None

        detection_system = self.detection_thread.detection_system

        world_coords = []

        # 获取缩放比例
        scale_x = float(self.scale_factor_X_input.text() or 1.0)
        scale_y = float(self.scale_factor_Y_input.text() or 1.0)

        rotation = self.coordinate_rotation  # 默认90度

        for point in pixel_points:
            if len(point) < 2:
                continue

            # 应用缩放比例
            scaled_x = point[0] * scale_x
            scaled_y = point[1] * scale_y

            # 转换为世界坐标
            world_x, world_y, success = detection_system.pixel_to_world_coords(
                scaled_x, scaled_y, rotation=rotation
            )
            if not success:
                continue

            # 应用用户偏移
            final_x = world_x + self.user_offset_x
            final_y = world_y + self.user_offset_y
            final_z = self.user_offset_z
            world_coords.append([final_x, final_y, final_z])

        return world_coords

    def save_history_paths(self):
        """保存历史路径到文件"""
        try:
            with open("history_paths.json", "w") as f:
                json.dump(self.history_paths, f)
        except Exception as e:
            print(f"保存历史路径失败: {str(e)}")

    def check_path_points(self, contour_points):
        """检查生成的路径点"""
        if not contour_points:
            print("没有生成路径点")
            return

        print(f"=== 路径点检查 ===")
        print(f"原始点数: {len(contour_points)}")

        # 计算路径边界
        x_points = [p[0] for p in contour_points]
        y_points = [p[1] for p in contour_points]
        min_x, max_x = min(x_points), max(x_points)
        min_y, max_y = min(y_points), max(y_points)

        print(f"X范围: {min_x:.2f} - {max_x:.2f}, 宽度: {max_x - min_x:.2f}")
        print(f"Y范围: {min_y:.2f} - {max_y:.2f}, 高度: {max_y - min_y:.2f}")

        # 计算周长
        perimeter = 0
        for i in range(len(contour_points)):
            p1 = contour_points[i]
            p2 = contour_points[(i + 1) % len(contour_points)]
            perimeter += math.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)

        print(f"周长: {perimeter:.2f}像素")


    def execute_all_points(self):
        """执行所有示教点"""
        if not self.teach_points:
            QMessageBox.warning(self, "无示教点", "没有可执行的示教点")
            return
        move_type = "MOVEJ" if self.move_type_combo.currentIndex() == 0 else "MOVEL"
        speed = self.speed_slider.value()
        self.execute_points(None, move_type, speed)

    def execute_selected_point(self):
        """执行选定的示教点"""
        point = self.get_selected_point()
        if not point:
            QMessageBox.warning(self, "未选择", "请先选择一个示教点")
            return
        move_type = "MOVEJ" if self.move_type_combo.currentIndex() == 0 else "MOVEL"
        speed = self.speed_slider.value()
        index = self.teach_points.index(point)
        self.execute_points([index], move_type, speed)

    def execute_points(self, point_indices=None, move_type='MOVEJ', speed=50):
        """执行示教点"""
        if not self.connection or not self.connection.is_connected():
            QMessageBox.warning(self, "未连接", "机械臂未连接，无法执行")
            return False
        if not self.teach_points:
            QMessageBox.warning(self, "无示教点", "没有可执行的示教点")
            return False

        if point_indices is None:
            points_to_execute = self.teach_points
        else:
            points_to_execute = [self.teach_points[i] for i in point_indices if 0 <= i < len(self.teach_points)]

        if not points_to_execute:
            QMessageBox.warning(self, "无效索引", "没有有效的示教点索引")
            return False

        self.execution_thread = threading.Thread(
            target=self.execute_points_thread,
            args=(points_to_execute, move_type, speed),
            daemon=True
        )
        self.execution_thread.start()
        print("执行线程已启动")

        self.progress_timer = QTimer()
        self.progress_timer.timeout.connect(self.update_progress)
        self.progress_timer.start(500)
        return True

    def execute_points_thread(self, points_to_execute, move_type, speed):
        """执行示教点的线程函数"""
        try:
            mc = self.connection.get_robot()
            if mc.focus_all_servos() != 1:
                print("警告：上电所有关节失败")

            print(f"开始执行 {len(points_to_execute)} 个点位 ({move_type}模式)...")
            self.execution_paused = False
            self.execution_stopped = False
            self.execution_progress = 0

            for i, point in enumerate(points_to_execute):
                print(f"\n执行点位 #{i + 1} ({point['name']})")
                self.execution_progress = int((i / len(points_to_execute)) * 100)

                if self.execution_stopped:
                    print("执行已被停止")
                    break

                while self.execution_paused:
                    print("执行暂停中...")
                    time.sleep(1)
                    if self.execution_stopped:
                        print("执行已被停止")
                        return False

                try:
                    target_position = None
                    if move_type == 'MOVEL':
                        target_position = point['coords']
                        result = mc.send_coords(target_position, speed, mode=1)
                        if result != 1:
                            print(f"发送坐标命令失败: {result}")
                            continue
                    else:
                        target_position = point.get('positions', point.get('angles', []))
                        result = mc.send_angles(target_position, speed)
                        if result != 1:
                            print(f"发送角度命令失败: {result}")
                            continue

                    start_time = time.time()
                    while True:
                        if not mc.is_moving():
                            break
                        if time.time() - start_time > 60:
                            print("警告：运动超时")
                            break
                        if self.execution_paused or self.execution_stopped:
                            mc.stop()
                            break
                        time.sleep(0.1)

                    if self.execution_paused:
                        print("运动已被暂停")
                        while self.execution_paused and not self.execution_stopped:
                            time.sleep(0.5)
                        if self.execution_stopped:
                            print("执行已被停止")
                            return False

                    if self.verification_enabled and target_position:
                        print("执行位置验证...")
                        if not self.verify_position(point, move_type):
                            print("警告：位置验证失败，可能需要重新执行")

                    self.last_executed_point = point
                    print(f"点位 #{i + 1} 执行完成")
                    time.sleep(0.5)
                except Exception as e:
                    print(f"执行点位 #{i + 1} 时出错: {str(e)}")
                    if not self.connection.is_connected():
                        print("尝试重新连接...")
                        self.connection.reconnect()
                    continue

            print("\n点位执行完毕" if not self.execution_stopped else "\n执行已停止")
            self.execution_progress = 100
            return True
        except Exception as e:
            print(f"执行过程中发生严重错误: {str(e)}")
            return False

    def update_progress(self):
        """更新进度条显示"""
        self.progress_bar.setValue(self.execution_progress)
        if self.execution_progress >= 100:
            self.progress_timer.stop()

    def verify_position(self, target_point, move_type):
        """验证当前位置是否达到目标位置"""
        try:
            mc = self.connection.get_robot()
            current_angles = mc.get_angles()
            current_coords = mc.get_coords()

            if not current_angles or not current_coords:
                print("验证失败：无法获取当前位置")
                return False

            if move_type == 'MOVEJ':
                target_angles = target_point.get('positions', target_point.get('angles', []))

                if len(current_angles) != 6 or len(target_angles) != 6:
                    print("验证失败：角度数据不完整")
                    return False

                errors = [abs(current_angles[i] - target_angles[i]) for i in range(6)]
                max_error = max(errors)

                if max_error > self.angle_tolerance:
                    print(f"验证失败：最大角度误差 {max_error:.2f}° > 容差 {self.angle_tolerance}°")
                    return False

                print(f"验证通过：最大角度误差 {max_error:.2f}°")
                return True
            else:
                current_pos = current_coords[:3]
                target_pos = target_point.get('coords', [])[:3]

                if len(current_pos) != 3 or len(target_pos) != 3:
                    print("验证失败：坐标数据不完整")
                    return False

                error = sum((current_pos[i] - target_pos[i]) ** 2 for i in range(3)) ** 0.5

                if error > self.coord_tolerance:
                    print(f"验证失败：位置误差 {error:.2f}mm > 容差 {self.coord_tolerance}mm")
                    return False

                print(f"验证通过：位置误差 {error:.2f}mm")
                return True
        except Exception as e:
            print(f"验证位置发生错误: {str(e)}")
            return False

    def pause_execution(self):
        """暂停执行"""
        if not self.execution_thread or not self.execution_thread.is_alive():
            QMessageBox.warning(self, "无执行", "没有正在执行的线程")
            return False

        try:
            mc = self.connection.get_robot()
            result = mc.pause()
            if result != 1:
                print(f"暂停命令失败: {result}")
            self.execution_paused = True
            print("执行已暂停")
            return True
        except Exception as e:
            QMessageBox.critical(self, "暂停失败", f"暂停执行失败: {str(e)}")
            return False

    def resume_execution(self):
        """恢复执行"""
        if not self.execution_thread or not self.execution_thread.is_alive():
            QMessageBox.warning(self, "无执行", "没有正在执行的线程")
            return False

        try:
            mc = self.connection.get_robot()
            result = mc.resume()
            if result != 1:
                print(f"恢复命令失败: {result}")
            self.execution_paused = False
            print("执行已恢复")
            return True
        except Exception as e:
            QMessageBox.critical(self, "恢复失败", f"恢复执行失败: {str(e)}")
            return False

    def stop_execution(self):
        """停止执行"""
        if not self.execution_thread or not self.execution_thread.is_alive():
            QMessageBox.warning(self, "无执行", "没有正在执行的线程")
            return False

        try:
            mc = self.connection.get_robot()
            result = mc.stop()
            if result != 1:
                print(f"停止命令失败: {result}")
            self.execution_paused = False
            self.execution_stopped = True
            print("执行已停止")
            return True
        except Exception as e:
            QMessageBox.critical(self, "停止失败", f"停止执行失败: {str(e)}")
            return False

    def closeEvent(self, event):
        """关闭窗口时停止所有线程"""
        if hasattr(self, 'camera_thread') and self.camera_thread and self.camera_thread.isRunning():
            self.camera_thread.stop()
            self.camera_thread = None

            # 关闭检测线程
        if self.detection_thread:
            self.detection_thread.stop()
            self.detection_thread = None

        if self.connection:
            self.connection.disconnect()
            self.connection.stop_heartbeat()
            self.stop_speech_recognition()

        event.accept()

    def stop_speech_recognition(self):
        """停止语音识别"""
        self.speech_recognition_active = False
        self.status_label.setText("语音识别已停止")
        print("语音识别停止")

    def change_camera_type(self, index):
        """切换摄像头类型"""
        self.camera_type = "network" if index == 1 else "local"
        print(f"切换摄像头类型为: {self.camera_type}")

    def load_onnx_model(self):
        """加载ONNX模型"""
        options = QFileDialog.Options()
        model_path, _ = QFileDialog.getOpenFileName(
            self, "选择ONNX模型文件", "",
            "ONNX模型 (*.onnx);;所有文件 (*)",
            options=options
        )

        if model_path:
            # 验证模型文件是否存在
            if not os.path.exists(model_path):
                QMessageBox.warning(self, "错误", "模型文件不存在")
                return

            success = self.CameraDetectionSystem.load_model(model_path)

            if success:
                self.onnx_model_path_input.setText(model_path)
                self.onnx_model_path = model_path
                QMessageBox.information(self, "模型加载", f"模型加载成功: {os.path.basename(model_path)}")
            else:
                QMessageBox.critical(self, "错误", "模型加载失败，请检查模型文件")

    # def toggle_detection(self):
    #     """切换目标检测状态"""
    #     if self.detect_button.text() == "启动检测":
    #         # 启动检测
    #         try:
    #             # 确保模型路径有效
    #             if not self.onnx_model_path:
    #                 QMessageBox.warning(self, "模型未加载", "请先加载ONNX模型")
    #                 return
    #
    #             # 创建新的检测线程（使用ONNX）
    #             self.detection_thread = ONNXDetectionThread(self.onnx_model_path)
    #
    #             # 修复5: 在启动线程前加载模型
    #             if not self.detection_thread.load_model():
    #                 QMessageBox.critical(self, "错误", "无法加载ONNX模型")
    #                 return
    #
    #             # 设置摄像头类型
    #             if self.camera_type == "network":
    #                 ip = self.camera_ip_input.text()
    #                 port = int(self.camera_port_input.text())
    #                 self.detection_thread.set_camera_type("network")
    #             else:
    #                 self.detection_thread.set_camera_type("local")
    #
    #             # 连接信号
    #             self.detection_thread.update_frame.connect(self.update_frame)
    #             self.detection_thread.detection_result.connect(self.update_detection_result)
    #             self.detection_thread.detection_coords.connect(self.handle_detection_coords)
    #
    #             self.detection_thread.start()
    #             self.detect_button.setText("停止检测")
    #             self.detect_button.setStyleSheet("background-color: #FF4D4D;")
    #         except Exception as e:
    #             QMessageBox.critical(self, "错误", f"无法启动目标检测: {str(e)}")
    #     else:
    #         # 停止检测
    #         if self.detection_thread:
    #             self.detection_thread.stop()
    #             self.detection_thread.wait(2000)  # 等待线程结束
    #             self.detection_thread = None
    #             self.detect_button.setText("启动检测")
    #             self.detect_button.setStyleSheet("")
    #             self.detection_label.clear()
    #             self.detection_label.setText("摄像头未启动")


    def handle_detection_coords(self, coords_list):
        """处理检测到的坐标"""
        for coords in coords_list:
            if coords:
                world_x, world_y = coords
                # print(f"检测到物体位置: 世界坐标({world_x:.1f}, {world_y:.1f})")


    def move_to_xyz(self):
        """移动到指定的XYZ坐标"""
        if not self.connection or not self.connection.is_connected():
            QMessageBox.warning(self, "未连接", "未连接到机器人，无法移动")
            return

        try:
            # 获取输入的坐标值
            x = float(self.target_x_input.text()) if self.target_x_input.text() else 0
            y = float(self.target_y_input.text()) if self.target_y_input.text() else 0
            z = float(self.target_z_input.text()) if self.target_z_input.text() else 0
            x += float(self.offset_x_input.text()) if self.offset_x_input.text() else 0
            y += float(self.offset_y_input.text()) if self.offset_y_input.text() else 0
            z += float(self.offset_z_input.text()) if self.offset_z_input.text() else 0

            # 获取当前姿态
            coords = self.connection.get_robot().get_coords()
            if len(coords) < 6:
                QMessageBox.warning(self, "错误", "无法获取机器人当前位置")
                return

            # 创建目标坐标
            target_coords = [x, y, z] + coords[3:]

            # 移动前记录4轴角度
            angles_before = self.connection.get_robot().get_angles()
            if angles_before and len(angles_before) >= 4:
                axis4_angle_before = angles_before[3]
                print(f"移动前记录的4轴角度: {axis4_angle_before}°")
            else:
                print("无法获取移动前角度")
                axis4_angle_before = None

            # 执行移动指令
            self.connection.get_robot().send_coords(target_coords, 30, mode=1)

            # 等待移动完成
            while self.connection.get_robot().is_moving():
                time.sleep(0.1)

            # 如果启用角度修正
            if self.angle_correction_checkbox.isChecked():
                # 移动完成后检测4轴角度
                angles_after = self.connection.get_robot().get_angles()

                if angles_after and len(angles_after) >= 4 and axis4_angle_before is not None:
                    axis4_angle_after = angles_after[3]
                    print(f"移动后4轴实际角度: {axis4_angle_after}°")

                    # 检查角度变化
                    angle_diff = abs(axis4_angle_before - axis4_angle_after)
                    if angle_diff > 0.7:
                        print(f"检测到4轴角度偏移 {angle_diff:.2f}°，进行修正...")

                        # 构建修正后的目标角度
                        correction_angles = list(angles_after)
                        correction_angles[3] = correction_angles[3] + 6.5  # 修正第4轴

                        # 执行角度修正
                        self.connection.get_robot().send_angles(correction_angles, 20)

                        # 验证修正结果
                        time.sleep(0.5)
                        final_angles = self.connection.get_robot().get_angles()
                        if len(final_angles) >= 4:
                            print(f"修正后4轴角度: {final_angles[3]}°")
                            print(f"最终误差: {abs(axis4_angle_before - final_angles[3]):.2f}°")
                    else:
                        print("4轴角度变化在允许范围内，无需修正")

            QMessageBox.information(self, "移动完成", f"已移动到位置: X={x}, Y={y}, Z={z}")

        except ValueError:
            QMessageBox.warning(self, "输入错误", "请输入有效的数字坐标")
        except Exception as e:
            QMessageBox.critical(self, "移动错误", f"移动过程中出错: {str(e)}")

    def detect_image(self):
        """检测单张图像"""
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择图像文件", "",
            "图像文件 (*.png *.jpg *.jpeg *.bmp);;所有文件 (*)",
            options=options
        )

        if not file_path:
            return

        # 创建新的检测线程用于单张图像
        image_detector = ONNXDetectionThread(self.onnx_model_path)

        #设置单张图像路径
        image_detector.set_single_image(file_path)

        # 连接信号
        image_detector.single_image_result.connect(self.handle_single_image_result)
        image_detector.finished.connect(lambda: image_detector.deleteLater())

        # 启动线程
        image_detector.start()

        # 显示加载状态
        self.detection_label.clear()
        self.detection_label.setText("正在检测图像...")
        self.detection_result_text.setText("处理中...")


    def calibrate_single_image(self):
        """使用单张图像进行标定"""
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择标定图像", "",
            "图像文件 (*.png *.jpg *.jpeg *.bmp);;所有文件 (*)",
            options=options
        )

        if not file_path:
            return

        # 确保检测线程已经创建
        if not self.detection_thread:
            self.create_detection_thread()

        # 执行标定
        if self.detection_thread:
            success = self.detection_thread.calibrate_single_image(file_path)
            if success:
                QMessageBox.information(self, "标定成功", "使用单张图像标定成功！")
            else:
                QMessageBox.warning(self, "标定失败", "无法完成标定，请确保图像中包含两个ArUco标记")

    def create_detection_thread(self):
        """创建检测线程"""
        try:
            # 获取摄像头类型和参数
            camera_type = "network" if self.camera_type_combo.currentIndex() == 1 else "local"
            ip = self.camera_ip_input.text()
            port = int(self.camera_port_input.text())

            # 创建检测线程，传入正确的参数
            self.detection_thread = ONNXDetectionThread(
                self.onnx_model_path,
                ip=ip,
                port=port
            )
            self.detection_thread.camera_type = camera_type

            # 连接信号
            self.detection_thread.update_frame.connect(self.update_frame)
            self.detection_thread.detection_result.connect(self.update_detection_result)
            self.detection_thread.detection_coords.connect(self.handle_detection_coords)

        except Exception as e:
            QMessageBox.critical(self, "错误", f"创建检测线程失败: {str(e)}")
            self.detection_thread = None


        # 确保检测线程已正确初始化
        if self.detection_thread and self.detection_thread.isRunning():
            try:
                self.detection_thread.perform_calibration()
            except Exception as e:
                QMessageBox.critical(self, "标定错误", f"标定过程中发生错误: {str(e)}")
        else:
            QMessageBox.warning(self, "错误", "无法启动标定，请先确保摄像头检测已正常启动")

    def apply_offsets(self, show_message=True):
        """应用用户设置的偏移量"""
        try:
            # 获取偏移值
            self.user_offset_x = float(self.offset_x_input.text() or 0)
            self.user_offset_y = float(self.offset_y_input.text() or 0)
            self.user_offset_z = float(self.offset_z_input.text() or 0)
            
            # 验证偏移值
            if abs(self.user_offset_x) > 200 or abs(self.user_offset_y) > 200 or abs(self.user_offset_z) > 300:
                QMessageBox.warning(self, "偏移过大", "偏移量不能超过±200mm,z不能超过300mm")
                return
            
            # 只在需要时显示消息框
            if show_message:
                QMessageBox.information(self, "偏移应用",
                                    f"偏移量已设置:\nX: {self.user_offset_x}mm\nY: {self.user_offset_y}mm\nZ: {self.user_offset_z}mm")
            
            return True
        except ValueError:
            if show_message:
                QMessageBox.warning(self, "输入错误", "请输入有效的数字偏移量")
        return False



    def toggle_speech_recognition(self):
        """切换语音识别状态"""
        if self.is_recording:
            # 停止录音
            self.is_recording = False
            self.speech_recognition_btn.setText('启动语音识别')
            self.speech_recognition_btn.setStyleSheet("""
                QPushButton {
                    background-color: #4CAF50;
                    color: white;
                    border-radius: 8px;
                }
            """)
            
            # 停止所有录音相关活动
            if self.recording_thread:
                self.recording_thread.stop_recording = True
            if self.recording_timer.isActive():
                self.recording_timer.stop()
            self.is_waiting_for_wake_word = False
            self.is_in_command_mode = False
            self.wake_word_detected = False
        else:
            # 开始语音识别
            self.is_recording = True
            self.speech_recognition_btn.setText('停止语音识别')
            self.speech_recognition_btn.setStyleSheet("""
                QPushButton {
                    background-color: #f44336;
                    color: white;
                    border-radius: 8px;
                }
            """)
            self.status_label.setText('等待唤醒词...')
            self.speak_response("已进入语音控制模式")
            self.speak_response("等待您的唤醒")
            
            # 进入唤醒词检测模式
            self.is_waiting_for_wake_word = True
            self.is_in_command_mode = False
            self.wake_word_detected = False
            
            # 开始3秒的唤醒词检测录音
            self.start_wake_word_recording()

    def start_wake_word_recording(self):
        """开始唤醒词检测录音"""
        if not self.is_recording:
            return
            
        print("开始唤醒词检测录音...")
        self.status_label.setText("唤醒词检测录音中...")
        
        # 创建录音线程，使用预校准的阈值
        self.recording_thread = RecordingThread(
            threshold=self.audio_threshold,
            max_duration=3.0,
            silence_duration=1.5
        )
        self.recording_thread.status_updated.connect(self.update_status)
        self.recording_thread.recording_finished.connect(self.process_wake_word_recording)
        self.recording_thread.start()
        
        # 设置超时定时器
        self.recording_timer.start(5000)  # 5秒

    def process_wake_word_recording(self, audio_file):
        """处理唤醒词检测录音"""
        if not self.is_recording:
            return
        
        if audio_file == '':
            print('唤醒词录音失败')
            self.status_label.setText("唤醒词录音失败")
            # 继续下一次唤醒词检测
            self.start_wake_word_recording()
            return
        
        try:
            print('正在检测唤醒词...')
            self.status_label.setText("正在检测唤醒词...")
            # 使用ASR模型识别语音
            user_input = self.asr_model(audio_file)
            print(f'识别结果: {user_input}')
            self.status_label.setText(f"识别结果: {user_input}")
            
            # 检查是否包含唤醒词
            if self.detect_wake_word(user_input):
                print(f"检测到唤醒词: {self.wake_word}")
                self.status_label.setText(f"检测到唤醒词: {self.wake_word}")
                self.wake_word_detected = True
                
                # 进入指令模式
                self.is_waiting_for_wake_word = False
                self.is_in_command_mode = True
                
                # 开始指令录音
                self.start_command_recording()
            else:
                print("未检测到唤醒词")
                self.status_label.setText("未检测到唤醒词，继续监听...")
                
                # 继续下一次唤醒词检测
                self.start_wake_word_recording()
        
        except Exception as e:
            print(f'唤醒词检测出错: {str(e)}')
            self.status_label.setText(f"唤醒词检测出错: {str(e)}")
            # 继续下一次唤醒词检测
            self.start_wake_word_recording()

    def start_command_recording(self):
        """开始指令录音（优化停顿处理）"""
        if not self.is_recording or not self.is_in_command_mode:
            return
            
        print("开始指令录音...")
        self.status_label.setText("请说出指令...")
        self.speak_response("等待您的指令")
        
        # 创建录音线程（增加静音检测时间）
        self.recording_thread = RecordingThread(
            threshold=self.audio_threshold,
            max_duration=20.0,  # 延长最大录音时间
            silence_duration=3.0  # 增加静音检测时间
        )
        self.recording_thread.status_updated.connect(self.update_status)
        self.recording_thread.recording_finished.connect(self.process_command_recording)
        self.recording_thread.start()
        
        # 设置超时定时器（比最大录音时间长）
        self.recording_timer.start(25000)  # 25秒


    def process_response(self, content):
        """处理LLM响应并播放"""
        # 移除思考标签并分句
        content = self.remove_think_tag(content)
        sentences = self.split_into_sentences(content)
        
        if not sentences:
            return
        
        print('=' * 40)
        print('开始文本转语音...')
        
        # 逐句生成语音并添加到播放队列
        for sentence in sentences:
            if sentence:  # 确保句子不为空
                audio_path = self.tts_model.run(sentence)
                if audio_path:
                    self.audio_system.add_to_queue(audio_path)
        
        print('语音已加入播放队列')
    
    def shutdown(self):
        """关闭系统"""
        self.audio_system.stop()

    # 分割文本成句子
    def split_into_sentences(self, text):
        """将文本分割成句子"""
        sentence_endings = r'(?<=[，。！？])|(?<=\.)'
        return [s.strip() for s in re.split(sentence_endings, text) if s.strip()]
    
    def remove_think_tag(self,text):
        return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)

    def update_recording_status(self):
        """更新录音状态显示"""
        # if self.is_recording:
        #     # 创建一个动态效果，显示录音正在进行
        #     dots = '.' * ((int(time.time() * 2) % 4) + 1)
        #     self.status_label.setText(f'录音中{dots}')
    
    def update_status(self, message):
        """更新状态信息"""
        print('User:',message)
    

    def process_command_recording(self, audio_file):
        """处理录音结果，支持参数修改"""
        try:
            # 检查录音文件是否有效
            if audio_file == '' or not os.path.exists(audio_file):
                print('录音失败或文件不存在')
                self.status_label.setText("录音失败")
                self.speak_response("录音失败，请重试")
                self.return_to_wake_word_mode()
                self.speak_response("录音失败，请重试")
                return
        
            print('正在识别语音...')
            self.status_label.setText("正在识别语音...")
            self.speak_response("正在识别您的指令")
            
            try:
                # 使用ASR模型识别语音
                user_input = self.asr_model(audio_file)
                print(f'识别结果: {user_input}')
                self.status_label.setText(f"识别结果: {user_input}")
                
                if not user_input or user_input.strip() == '':
                    print('未识别到有效语音')
                    self.status_label.setText("未识别到有效语音")
                    self.speak_response("未识别到有效语音，请重试")
                    self.return_to_wake_word_mode()
                    self.speak_response("抱歉，我没有听清")
                    return
                
            # 提取参数
                parameters = self.extract_parameters(user_input)
                print(f"提取到的参数: {parameters}")
                self.speak_response("正在分析您的指令")
                # 在原有的参数提取后，添加对新指令的处理
                if 'command' in parameters:
                    command = parameters['command']
                    value = parameters.get('value')
                    
                    # 处理特定类型的指令

                    if command in ["OPEN_CAMERA", "CLOSE_CAMERA", "CONNECT_ROBOT", "DISCONNECT_ROBOT",
                                "START_CALIBRATION", "START_GRINDING", "STOP_GRINDING",
                                "START_SPEECH_RECOGNITION", "STOP_SPEECH_RECOGNITION",
                                "MOTOR_START", "MOTOR_STOP", "MOTOR_FORWARD", "MOTOR_REVERSE",
                                "MOTOR_EMERGENCY_STOP", "MOVE_TO_SAFE_POSITION"]:
                        # 执行不需要额外参数的指令
                        self.handle_single_parameter_command(command, value)
                        
                        # 删除临时文件并返回唤醒词模式
                        try:
                            os.remove(audio_file)
                        except:
                            pass
                        self.return_to_wake_word_mode()
                        return
                    
                    elif command in ["MOTOR_SET_SPEED", "MOVE_TO_TEACH_POINT", "MOVE_TO_COORDINATES"]:
                        # 执行需要参数的指令
                        if value is not None:
                            self.handle_single_parameter_command(command, value)
                        else:
                            self.speak_response("未提供必要的参数值")
                        
                        # 删除临时文件并返回唤醒词模式
                        try:
                            os.remove(audio_file)
                        except:
                            pass
                        self.return_to_wake_word_mode()
                        return
                
                # 处理参数修改指令
                if parameters:
                    # 应用参数修改
                    if self.apply_voice_parameters(parameters):
                        # 生成反馈消息
                        feedback = "参数修改成功"
                        if 'offset_x' in parameters:
                            feedback += f"，X偏移设置为{parameters['offset_x']}毫米"
                        if 'offset_y' in parameters:
                            feedback += f"，Y偏移设置为{parameters['offset_y']}毫米"
                        if 'offset_z' in parameters:
                            feedback += f"，Z偏移设置为{parameters['offset_z']}毫米"
                        
                        # 语音反馈
                        self.speak_response(feedback)
                        
                        # 删除临时文件并返回唤醒词模式
                        try:
                            os.remove(audio_file)
                        except:
                            pass
                        self.return_to_wake_word_mode()
                        return
                
                # 2. 如果没有直接提取到参数，使用LLM进行判断
                # print('使用LLM分析指令...')
                # self.status_label.setText("使用AI分析指令...")
                # self.speak_response("正在分析您的指令")
                
                # full_response = ""
                # detected_command = None
                # llm_parameters = {}
                
                # # 流式获取响应
                # for chunk in self.llm_model.generate(user_input):
                #     print(chunk, end='', flush=True)
                #     full_response += chunk
                    
                #     # 检查是否包含指令标记
                #     detected_command = self.detect_command_in_response(full_response)
                #     if detected_command:
                #         # 安全提取LLM参数
                #         try:
                #             llm_parameters = self.extract_parameters(full_response)
                #         except Exception as e:
                #             print(f"LLM参数提取出错: {str(e)}")
                #             llm_parameters = {}
                #         break
                
                # print('\nLLM分析完成')
                
                # 3. 处理检测到的指令和参数
                # if detected_command:
                #     print(f"检测到指令: {detected_command}")
                #     self.status_label.setText(f"检测到指令: {detected_command}")
                #     self.speak_response(f"检测到指令: {detected_command}")
                    
                #     # 定义单独参数指令映射
                #     single_commands = {
                #         "MODIFY_X_OFFSET": 'offset_x',
                #         "MODIFY_Y_OFFSET": 'offset_y',
                #         "MODIFY_Z_OFFSET": 'offset_z',
                #         "MODIFY_X_STEP": 'grinding_x_step',
                #         "MODIFY_Y_STEP": 'grinding_y_step',
                #         "MODIFY_Z_STEP": 'grinding_z_step',
                #         "MODIFY_LOOPS": 'loops',
                #         "MODIFY_SPEED": 'speed'
                #     }
                    
                #     # 检查是否是单独参数指令
                #     if detected_command in single_commands:
                #         param_key = single_commands[detected_command]
                #         param_value = None
                        
                #         # 优先使用LLM提取的参数
                #         if param_key in llm_parameters:
                #             param_value = llm_parameters[param_key]
                #         # 其次使用关键词提取的参数
                #         elif param_key in validated_params:
                #             param_value = validated_params[param_key]
                #         # 最后使用语音中的数字（取第一个）
                #         elif numbers:
                #             param_value = numbers[0]
                        
                #         if param_value is not None:
                #             # 处理单独参数指令
                #             self.handle_single_parameter_command(detected_command, param_value)
                #         else:
                #             print(f"未找到{param_key}的有效数值")
                #             self.speak_response(f"未找到有效数值，请重新设置")
                #     else:
                #         # 处理复合参数指令
                #         if llm_parameters:
                #             print(f"从LLM响应中提取到参数: {llm_parameters}")
                            
                #             # 验证并应用LLM参数
                #             llm_validated = {}
                #             for key, value in llm_parameters.items():
                #                 if 'offset' in key:
                #                     llm_validated[key] = validate_parameter(key, value, -50, 50)
                #                 elif 'step' in key:
                #                     llm_validated[key] = validate_parameter(key, value, -10, 10)
                #                 elif key == 'loops':
                #                     llm_validated[key] = validate_parameter(key, value, 1, 100)
                #                 elif key == 'speed':
                #                     llm_validated[key] = validate_parameter(key, value, 1, 100)
                #                 else:
                #                     llm_validated[key] = value
                            
                #             self.apply_voice_parameters(llm_validated)
                        
                #         # 处理指令
                #         self.process_command(detected_command)
                
                # # 4. 处理回复内容
                # if full_response.strip():
                #     clean_response = self.remove_think_tag(full_response)
                #     print(f"AI回复: {clean_response.lstrip()}")
                #     self.status_label.setText(f"AI回复: {clean_response}")
                    
                #     # 语音播报回复内容
                #     self.speak_response(clean_response)
                # else:
                #     self.speak_response("指令已处理")
                
                # 回到唤醒词检测模式
                self.return_to_wake_word_mode()
                
            except Exception as e:
                print(f'处理过程中出错: {str(e)}')
                self.status_label.setText(f"处理出错: {str(e)}")
                self.speak_response("处理指令时出错")
                import traceback
                traceback.print_exc()
                # 确保回到唤醒词模式
                self.return_to_wake_word_mode()
        
        except Exception as e:
            print(f'处理过程中发生严重错误: {str(e)}')
            self.status_label.setText(f"严重错误: {str(e)}")
            self.speak_response("系统发生错误，请检查日志")
            import traceback
            traceback.print_exc()
        finally:
        # 确保删除临时文件
            try:
                if os.path.exists(audio_file):
                    os.remove(audio_file)
            except:
                pass

    def return_to_wake_word_mode(self):
        """返回到唤醒词检测模式"""
        try:
            self.is_waiting_for_wake_word = True
            self.is_in_command_mode = False
            self.wake_word_detected = False
            self.status_label.setText('等待唤醒词...')
            self.speak_response("等待您的唤醒")
            self.start_wake_word_recording()
        except Exception as e:
            print(f"返回唤醒模式出错: {str(e)}")
            # 尝试重置状态
            self.is_waiting_for_wake_word = True
            self.is_in_command_mode = False
            self.wake_word_detected = False
            self.status_label.setText('系统准备就绪')

    def handle_single_parameter_command(self, command, value):
        """处理单独参数修改指令"""
        try:
            self.speak_response("正在处理您的请求")
            # 根据指令类型应用参数
            feedback = ""
            if command == "MODIFY_X_OFFSET":
                self.offset_x_input.setText(str(value))
                self.apply_offsets()
                feedback = f"X偏移已设置为{value}毫米"
            elif command == "MODIFY_Y_OFFSET":
                self.offset_y_input.setText(str(value))
                self.apply_offsets()
                feedback = f"Y偏移已设置为{value}毫米"
            elif command == "MODIFY_Z_OFFSET":
                self.offset_z_input.setText(str(value))
                self.apply_offsets()
                feedback = f"Z偏移已设置为{value}毫米"
            elif command == "MODIFY_X_STEP":
                self.x_step_input.setText(str(value))
                self.apply_grinding_params()
                feedback = f"X进深已设置为{value}毫米"
            elif command == "MODIFY_Y_STEP":
                self.y_step_input.setText(str(value))
                self.apply_grinding_params()
                feedback = f"Y进深已设置为{value}毫米"
            elif command == "MODIFY_Z_STEP":
                self.z_step_input.setText(str(value))
                self.apply_grinding_params()
                feedback = f"Z进深已设置为{value}毫米"
            elif command == "MODIFY_LOOPS":
                # 确保循环次数是整数
                value = int(value) if isinstance(value, float) else value
                self.loop_count_input.setText(str(value))
                self.apply_grinding_params()
                feedback = f"循环次数已设置为{value}次"
            elif command == "MODIFY_SPEED":
                # 确保速度在1-100范围内
                value = max(1, min(100, int(value)))
                self.speed_slider.setValue(value)
                feedback = f"速度已设置为{value}"
            # 新增指令：摄像头控制
            elif command == "OPEN_CAMERA":
                self.toggle_camera()
                feedback = "摄像头已开启"
            elif command == "CLOSE_CAMERA":
                self.close_camera()
                feedback = "摄像头已关闭"
            # 新增指令：机器人连接控制
            elif command == "CONNECT_ROBOT":
                self.toggle_connection()
                feedback = "机器人已连接"
            elif command == "DISCONNECT_ROBOT":
                if self.connection and self.connection.is_connected():
                    self.connection.disconnect()
                    self.update_ui_state(False)
                feedback = "机器人已断开连接"
            # 新增指令：标定控制
            elif command == "START_CALIBRATION":
                self.start_calibration()
                feedback = "开始标定"
            # 新增指令：打磨控制
            elif command == "START_GRINDING":
                self.toggle_grinding()
                feedback = "开始打磨"
            elif command == "STOP_GRINDING":
                self.toggle_grinding()
                feedback = "停止打磨"
            # 新增指令：语音识别控制
            elif command == "START_SPEECH_RECOGNITION":
                self.toggle_speech_recognition()
                feedback = "语音识别已开启"
            elif command == "STOP_SPEECH_RECOGNITION":
                self.toggle_speech_recognition()
                feedback = "语音识别已关闭"
            # 新增指令：电机控制
            elif command == "MOTOR_START":
                if hasattr(self, 'motor_controller') and self.motor_controller:
                    self.motor_controller.start()
                feedback = "电机已启动"
            elif command == "MOTOR_STOP":
                if hasattr(self, 'motor_controller') and self.motor_controller:
                    self.motor_controller.stop()
                feedback = "电机已停止"
            elif command == "MOTOR_FORWARD":
                if hasattr(self, 'motor_controller') and self.motor_controller:
                    self.motor_controller.forward()
                feedback = "电机正转"
            elif command == "MOTOR_REVERSE":
                if hasattr(self, 'motor_controller') and self.motor_controller:
                    self.motor_controller.reverse()
                feedback = "电机反转"
            elif command == "MOTOR_EMERGENCY_STOP":
                if hasattr(self, 'motor_controller') and self.motor_controller:
                    self.motor_controller.emergency_stop()
                feedback = "电机急停"
            elif command == "MOTOR_SET_SPEED":
                if hasattr(self, 'motor_controller') and self.motor_controller:
                    self.motor_controller.set_speed(value)
                feedback = f"电机速度设置为{value}"
            # 新增指令：机械臂安全位置
            elif command == "MOVE_TO_SAFE_POSITION":
                self.move_to_safe_position()
                feedback = "机械臂已移动到安全位置"
            # 新增指令：移动到指定示教点
            elif command == "MOVE_TO_TEACH_POINT":
                point_name = str(value)  # 这里value应该是示教点名称
                self.move_to_named_teach_point(point_name)
                feedback = f"机械臂已移动到{point_name}"
            # 新增指令：移动到指定坐标
            elif command == "MOVE_TO_COORDINATES":
                # 这里value应该是一个包含坐标的字符串，如"100,200,300"
                coords = [float(coord.strip()) for coord in str(value).split(',')]
                if len(coords) >= 3:
                    self.move_to_xyz_coords(coords[0], coords[1], coords[2])
                    feedback = f"机械臂已移动到坐标({coords[0]}, {coords[1]}, {coords[2]})"
                else:
                    feedback = "坐标格式错误，请使用x,y,z格式"
            
            if feedback:
                self.status_label.setText(feedback)
                self.speak_response(feedback)
            self.speak_response("指令执行成功")
            return True
        except Exception as e:
            error_msg = f"参数修改失败: {str(e)}"
            print(f"{error_msg}: {str(e)}")
            self.status_label.setText(error_msg)
            self.speak_response(error_msg)
            return False
        
    def move_to_safe_position(self):
        """移动机械臂到安全位置"""
        if not self.connection or not self.connection.is_connected():
            QMessageBox.warning(self, "未连接", "未连接到机器人，无法移动")
            return
        
        try:
            # 安全位置坐标或角度
            safe_position = [0, 0, 0, 0, 0, 0] 
            self.connection.get_robot().send_angles(safe_position, 50)
            QMessageBox.information(self, "移动", "机械臂正在移动到安全位置")
        except Exception as e:
            QMessageBox.critical(self, "移动错误", f"移动过程中出错: {str(e)}")

    def move_to_named_teach_point(self, point_name):
        """移动到指定名称的示教点"""
        point = None
        for teach_point in self.teach_points:
            if teach_point.get('name') == point_name:
                point = teach_point
                break
        
        if point:
            self.move_to_teach_point(point)
        else:
            QMessageBox.warning(self, "未找到", f"未找到名为'{point_name}'的示教点")

    def move_to_xyz_coords(self, x, y, z):
        """移动到指定的XYZ坐标"""
        if not self.connection or not self.connection.is_connected():
            QMessageBox.warning(self, "未连接", "未连接到机器人，无法移动")
            return
        
        try:
            # 获取当前姿态
            coords = self.connection.get_robot().get_coords()
            if len(coords) < 6:
                QMessageBox.warning(self, "错误", "无法获取机器人当前位置")
                return
            
            # 创建目标坐标
            target_coords = [x, y, z] + coords[3:]
            self.connection.get_robot().send_coords(target_coords, 30, mode=1)
            QMessageBox.information(self, "移动", f"机械臂正在移动到坐标({x}, {y}, {z})")
        except Exception as e:
            QMessageBox.critical(self, "移动错误", f"移动过程中出错: {str(e)}")
        
    def speak_response(self, text):
        """播放语音响应"""
        self.audio_system.add_to_queue(text)
    
    def system_start(self):
        """系统启动时播放预加载语音"""
        self.audio_system.add_to_queue("system_start")
        
    def custom_message(self, text):
        """播放自定义消息（动态生成）"""
        self.audio_system.add_to_queue(text)


    def detect_direct_command(self, user_input):
        """直接关键词匹配指令"""
        user_input = user_input.lower()
        
        command_map = {
            "启动": "START_GRINDING",
            "开始": "START_GRINDING",
            "停止": "STOP_GRINDING",
            "结束": "STOP_GRINDING",
            "标定": "START_CALIBRATION",
            "校准": "START_CALIBRATION",
            "连接机器人": "CONNECT_ROBOT",
            "断开机器人": "DISCONNECT_ROBOT",
            "连接麦克风": "CONNECT_MICROPHONE",
            "断开麦克风": "DISCONNECT_MICROPHONE",
            "启动语音识别": "START_SPEECH_RECOGNITION",
            "停止语音识别": "STOP_SPEECH_RECOGNITION"
        }
        
        # 检查是否完全匹配某个指令
        for keyword, command in command_map.items():
            if keyword in user_input:
                return command
        
        return None

    def detect_command_in_response(self, response):
        """从响应中检测指令"""
        # 方法1：正则匹配
        match = re.search(r'<command>(.*?)</command>', response, re.IGNORECASE)
        if match:
            return match.group(1)
        
        # 方法2：关键词匹配（备用）
        command_keywords = {
            "START_GRINDING": ["启动打磨", "开始打磨"],
            "STOP_GRINDING": ["停止打磨", "结束打磨"],
            "START_CALIBRATION": ["开始标定", "进行校准"],
            "CONNECT_ROBOT": ["连接机器人", "机器人连接"],
            "DISCONNECT_ROBOT": ["断开机器人", "机器人断开"],
            "CONNECT_MICROPHONE": ["连接麦克风", "麦克风连接"],
            "DISCONNECT_MICROPHONE": ["断开麦克风", "麦克风断开"],
            "START_SPEECH_RECOGNITION": ["启动语音识别", "语音识别开启"],
            "STOP_SPEECH_RECOGNITION": ["停止语音识别", "语音识别关闭"]
        }
        
        for command, keywords in command_keywords.items():
            for keyword in keywords:
                if keyword in response:
                    return command
        
        return None
    
    def process_command(self, command):
        """处理从LLM返回的指令"""
        print(f"执行指令: {command}")
        self.status_label.setText(f"执行指令: {command}")
        
        # 根据指令调用对应函数
        if command == "START_GRINDING":
            self.toggle_grinding()
        elif command == "STOP_GRINDING":
            self.toggle_grinding()  # 停止和启动可能是同一个按钮
        elif command == "START_CALIBRATION":
            self.start_calibration()
        elif command == "CONNECT_ROBOT":
            self.toggle_connection()  # 直接调用连接/断开方法
        elif command == "DISCONNECT_ROBOT":
            self.toggle_connection()  # 断开连接也是同一个方法
        elif command == "START_SPEECH_RECOGNITION":
            self.toggle_speech_recognition()
        elif command == "STOP_SPEECH_RECOGNITION":
            self.toggle_speech_recognition()

        elif command == "MODIFY_OFFSET":
            # 这里可以添加特定的偏移修改逻辑
            self.status_label.setText("坐标偏移已修改")
        elif command == "MODIFY_GRINDING_PARAMS":
            self.status_label.setText("打磨参数已修改")
        elif command == "MODIFY_LOOPS":
            self.status_label.setText("循环次数已修改")
        elif command == "MODIFY_SPEED":
            self.status_label.setText("速度已修改")

    def extract_parameters(self, text):
        """使用优化的中文数字处理方案"""
            # 初始化结果字典
        result = {}
        text_lower = text.lower().strip()
        
        # === 1. 处理特殊指令===
        # 摄像头控制
        if any(word in text_lower for word in ["打开摄像头", "启动摄像头", "开启摄像头"]):
            return {"command": "OPEN_CAMERA"}
        elif any(word in text_lower for word in ["关闭摄像头", "停止摄像头", "关掉摄像头"]):
            return {"command": "CLOSE_CAMERA"}
        
        # 机器人连接控制
        elif any(word in text_lower for word in ["连接机器人", "连接机械臂", "机器人连接"]):
            return {"command": "CONNECT_ROBOT"}
        elif any(word in text_lower for word in ["断开机器人", "断开连接", "机器人断开"]):
            return {"command": "DISCONNECT_ROBOT"}
        
        # 标定控制
        elif any(word in text_lower for word in ["开始标定", "进行标定", "校准", "标定摄像头"]):
            return {"command": "START_CALIBRATION"}
        
        # 打磨控制
        elif any(word in text_lower for word in ["开始打磨", "启动打磨", "运行打磨"]):
            return {"command": "START_GRINDING"}
        elif any(word in text_lower for word in ["停止打磨", "结束打磨", "暂停打磨"]):
            return {"command": "STOP_GRINDING"}
        
        # 语音识别控制
        elif any(word in text_lower for word in ["开启语音识别", "启动语音识别", "语音控制开启"]):
            return {"command": "START_SPEECH_RECOGNITION"}
        elif any(word in text_lower for word in ["关闭语音识别", "停止语音识别", "语音控制关闭"]):
            return {"command": "STOP_SPEECH_RECOGNITION"}
        
        # 电机控制
        elif any(word in text_lower for word in ["启动电机", "开启电机", "电机启动"]):
            return {"command": "MOTOR_START"}
        elif any(word in text_lower for word in ["停止电机", "关闭电机", "电机关闭"]):
            return {"command": "MOTOR_STOP"}
        elif any(word in text_lower for word in ["电机正转", "正转", "电机向前"]):
            return {"command": "MOTOR_FORWARD"}
        elif any(word in text_lower for word in ["电机反转", "反转", "电机向后"]):
            return {"command": "MOTOR_REVERSE"}
        elif any(word in text_lower for word in ["电机急停", "紧急停止", "急停"]):
            return {"command": "MOTOR_EMERGENCY_STOP"}
        
        # 机械臂安全位置
        elif any(word in text_lower for word in ["安全位置", "回到原点", "归位", "返回零点"]):
            return {"command": "MOVE_TO_SAFE_POSITION"}
        
        # === 2. 处理带参数的指令 ===
        # 设置电机速度
        if "电机速度" in text_lower or "转速" in text_lower:
            numbers = re.findall(r'\d+', text_lower)
            if numbers:
                try:
                    speed = int(numbers[0])
                    return {"command": "MOTOR_SET_SPEED", "value": speed}
                except ValueError:
                    pass
        
        # 设置X/Y/Z偏移
        if "偏移" in text_lower or "偏一" in text_lower:
            # 提取轴信息
            axis = None
            if "x" in text_lower or "叉" in text_lower:
                axis = "x"
            elif "y" in text_lower or "y" in text_lower:
                axis = "y"
            elif "z" in text_lower or "z" in text_lower:
                axis = "z"
            
            # 提取数值
            numbers = re.findall(r'[-+]?\d*\.?\d+', text_lower)
            if numbers:
                try:
                    value = float(numbers[0])
                    return {"command": f"MODIFY_{axis.upper()}_OFFSET", "value": value}
                except (ValueError, TypeError):
                    pass
        
        # 设置循环次数
        if "循环" in text_lower or "次数" in text_lower:
            numbers = re.findall(r'\d+', text_lower)
            if numbers:
                try:
                    loops = int(numbers[0])
                    return {"command": "MODIFY_LOOPS", "value": loops}
                except ValueError:
                    pass
        
        # 设置速度
        if "速度" in text_lower and ("打磨" not in text_lower and "电机" not in text_lower):
            numbers = re.findall(r'\d+', text_lower)
            if numbers:
                try:
                    speed = int(numbers[0])
                    return {"command": "MODIFY_SPEED", "value": speed}
                except ValueError:
                    pass
        
        # 移动到指定示教点
        if "移动到" in text_lower and ("点" in text_lower or "位置" in text_lower):
            # 提取点名称
            point_match = re.search(r'移动到(.+?)(点|位置)', text_lower)
            if point_match:
                point_name = point_match.group(1).strip()
                return {"command": "MOVE_TO_TEACH_POINT", "value": point_name}
        
        # 移动到指定坐标
        if "移动到" in text_lower and ("坐标" in text_lower or "位置" in text_lower):
            # 提取坐标值
            coords = re.findall(r'[-+]?\d*\.?\d+', text_lower)
            if len(coords) >= 3:
                try:
                    x = float(coords[0])
                    y = float(coords[1])
                    z = float(coords[2])
                    return {"command": "MOVE_TO_COORDINATES", "value": [x, y, z]}
                except ValueError:
                    pass
            elif len(coords) == 1:
                # 可能是单独指定Z高度
                try:
                    z = float(coords[0])
                    return {"command": "MODIFY_Z_OFFSET", "value": z}
                except ValueError:
                    pass

        parameters = {}
        text_lower = text.lower()
        print(f"[DEBUG] 原始识别结果: {text_lower}")
        
        # === 口语化表达标准化 ===
        text_lower = self.normalize_colloquial_expressions(text_lower)
        print(f"[DEBUG] 标准化后的文本: {text_lower}")
        
        # === 容错关键词映射 ===
        keyword_map = {
            '偏移': ['偏移', '片以', '偏移量', '偏一', '片移', '偏移值'],
            'x': ['x', '叉', 'x轴', 'x方向'],
            'y': ['y', 'y', 'y轴', 'y方向'],
            'z': ['z', 'z', 'z轴', 'z方向'],
            '轴': ['轴', '则', '周', '州', '坐', '作', '逐', '追逐'],  # 添加更多可能的误识别
            '坐标': ['坐标', '做标', '坐表', '作表']  # 添加坐标的映射
        }
        
        # 替换识别错误的关键词
        normalized_text = text_lower
        for correct_word, variants in keyword_map.items():
            for variant in variants:
                if variant in normalized_text:
                    normalized_text = normalized_text.replace(variant, correct_word)
                    print(f"[DEBUG] 替换关键词: {variant} -> {correct_word}")
        
        print(f"[DEBUG] 规范化后的文本: {normalized_text}")
        
        # === 使用cn2an严格模式转换中文数字 ===
        # 先提取所有中文数字片段
        chinese_digits = '零一二两三四五六七八九十百千万'
        number_pattern = r'([{}]+)'.format(chinese_digits)
        matches = re.findall(number_pattern, normalized_text)
        
        processed_text = normalized_text
        for match in matches:
            try:
                # 使用严格模式转换中文数字
                arabic_num = cn2an.cn2an(match, "strict")
                processed_text = processed_text.replace(match, str(arabic_num), 1)
                print(f"[DEBUG] 转换中文数字: {match} -> {arabic_num}")
            except Exception as e:
                print(f"[WARNING] 中文数字转换失败: {match} - {str(e)}")
        
        print(f"[DEBUG] 处理后的文本: {processed_text}")
        
        # === 精确参数提取 ===
        param_patterns = {
            'offset_x': r'(?:x|X)[\s：:]*[轴]?[向位]?偏移[\s：:]*([-+]?\d*\.?\d+)',
            'offset_y': r'(?:y|Y)[\s：:]*[轴]?[向位]?偏移[\s：:]*([-+]?\d*\.?\d+)',
            'offset_z': r'(?:z|Z)[\s：:]*[轴]?[向位]?偏移[\s：:]*([-+]?\d*\.?\d+)'
        }
        
        for param, pattern in param_patterns.items():
            matches = re.findall(pattern, processed_text)
            if matches:
                try:
                    parameters[param] = float(matches[0])
                    print(f"[DEBUG] 精确匹配参数: {param}={matches[0]}")
                except ValueError:
                    pass
        
        if not parameters:
            # 检测轴关键词
            axis_detected = None
            if re.search(r'(?:x|X)[\s：:]*[轴]?', processed_text):
                axis_detected = 'x'
            elif re.search(r'(?:y|Y)[\s：:]*[轴]?', processed_text):
                axis_detected = 'y'
            elif re.search(r'(?:z|Z)[\s：:]*[轴]?', processed_text):
                axis_detected = 'z'
            
            # 检测偏移关键词
            offset_detected = re.search(r'偏移', processed_text) is not None
            
            # 提取所有数字
            numbers = []
            number_pattern = r'[-+]?\d*\.?\d+'
            number_strings = re.findall(number_pattern, processed_text)
            
            for num_str in number_strings:
                try:
                    numbers.append(float(num_str))
                except ValueError:
                    continue
            
            print(f"[DEBUG] 后备数字列表: {numbers}")
            
            # 智能分配参数
            if offset_detected and axis_detected and numbers:
                parameters[f'offset_{axis_detected}'] = numbers[0]
                print(f"[DEBUG] 智能分配: offset_{axis_detected}={numbers[0]}")
            elif offset_detected and numbers:
                # 默认分配到Z轴
                parameters['offset_z'] = numbers[0]
                print(f"[DEBUG] 默认分配: offset_z={numbers[0]}")
        
        print(f"[DEBUG] 最终提取参数: {parameters}")
        return parameters
    
    def normalize_colloquial_expressions(self, text):
        """将口语化表达转换为标准中文数字"""
        # 口语化数字转换
        colloquial_map = {
            '两': '二',  # 两百 -> 二百
            '俩': '二',  # 俩百 -> 二百
            '仨': '三',  # 仨百 -> 三百
            '廿': '二十', # 廿一 -> 二十一
            '卅': '三十', # 卅一 -> 三十一
            '卌': '四十'  # 卌一 -> 四十一
        }
        
        # 口语化量词转换
        measure_word_map = {
            '个': '',
            '只': '',
            '条': '',
            '张': '',
            '块': '',
            '枚': ''
        }
        
        # 替换口语化数字
        for colloquial, standard in colloquial_map.items():
            text = text.replace(colloquial, standard)
        
        # 替换口语化量词
        for measure_word in measure_word_map:
            text = text.replace(measure_word, measure_word_map[measure_word])
        
        return text
    
    def apply_voice_parameters(self, parameters):
        """应用从语音中提取的参数"""
        try:
            print(f"[DEBUG] 应用语音参数: {parameters}")
            
            # 确保使用提取的参数
            if not parameters:
                print("[WARNING] 没有提取到参数")
                self.speak_response("未识别到有效参数")
                return False
            
            # 更新UI显示
            if 'offset_x' in parameters:
                self.offset_x_input.setText(str(parameters['offset_x']))
                self.user_offset_x = parameters['offset_x']
            if 'offset_y' in parameters:
                self.offset_y_input.setText(str(parameters['offset_y']))
                self.user_offset_y = parameters['offset_y']
            if 'offset_z' in parameters:
                self.offset_z_input.setText(str(parameters['offset_z']))
                self.user_offset_z = parameters['offset_z']
            
            # 生成反馈消息
            feedback = "参数修改成功"
            if 'offset_x' in parameters:
                feedback += f"，X偏移设置为{parameters['offset_x']}毫米"
            if 'offset_y' in parameters:
                feedback += f"，Y偏移设置为{parameters['offset_y']}毫米"
            if 'offset_z' in parameters:
                feedback += f"，Z偏移设置为{parameters['offset_z']}毫米"
            
            # 语音反馈
            self.speak_response(feedback)
            
            return True
        except Exception as e:
            print(f"[ERROR] 应用参数失败: {str(e)}")
            self.speak_response("参数应用失败")
            return False
    
    def detect_wake_word(self, text):
        """使用拼音相似度检测唤醒词"""
        # 唤醒词列表
        wake_words = ["小智", "小志", "小制", "小知", "小之", "小直", "小纸"]
        
        # 将文本转换为拼音
        text_pinyin = ''.join(lazy_pinyin(text, style=Style.NORMAL))
        
        # 检查每个唤醒词的拼音相似度
        for word in wake_words:
            word_pinyin = ''.join(lazy_pinyin(word, style=Style.NORMAL))
            
            # 简单相似度检查
            if word_pinyin in text_pinyin:
                return True
            if len(word_pinyin) > 2 and text_pinyin.startswith(word_pinyin[:2]):
                return True
        return False










































 
    def toggle_camera(self):
        """切换摄像头状态"""
        if not hasattr(self, 'camera_thread') or not self.camera_thread or not self.camera_thread.isRunning():
            # 获取摄像头类型和参数
            camera_type = "network" if self.camera_type_combo.currentIndex() == 1 else "local"
            ip = self.camera_ip_input.text()
            port = int(self.camera_port_input.text())

            # 创建摄像头线程
            self.camera_thread = CameraThread(camera_type, ip, port)
            self.camera_thread.update_frame.connect(self.update_frame)
            self.camera_thread.status_changed.connect(self.update_camera_status)
            self.camera_thread.start()

            self.open_camera_btn.setText("摄像头运行中")
            self.open_camera_btn.setEnabled(False)
            self.close_camera_btn.setEnabled(True)
        else:
            QMessageBox.information(self, "摄像头状态", "摄像头已在运行中")

    def close_camera(self):
        """关闭摄像头"""
        if hasattr(self, 'camera_thread') and self.camera_thread and self.camera_thread.isRunning():
            self.camera_thread.stop()
            self.camera_thread = None

            # 清空画面
            self.detection_label.clear()
            self.detection_label.setText("摄像头已关闭")

            self.open_camera_btn.setText("打开摄像头")
            self.open_camera_btn.setEnabled(True)
            self.close_camera_btn.setEnabled(False)
            self.update_camera_status("摄像头: 关闭")

    def update_camera_status(self, message):
        """更新摄像头状态"""
        self.camera_status_label.setText(message)
        if "摄像头: " not in message:
            self.camera_status_label.setText(f"摄像头: {message}")

    def handle_task_completed(self, task_id, result):
        """处理任务完成"""
        print(f"任务 {task_id} 完成，结果类型: {type(result)}")

        # 根据任务ID执行相应的后续操作
        if task_id == self.task_ids.get('camera'):
            self.update_camera_frame(result)
        elif task_id == self.task_ids.get('point_cloud'):
            # 点云数据获取任务完成
            self._handle_point_cloud_completed(result)
            # 清理任务ID
            self.task_ids['point_cloud'] = None
        elif task_id == getattr(self, 'current_grinding_task_id', None):
            # 打磨任务完成
            print("打磨任务完成")
            # 在主线程中安全更新UI
            self.grinding_button.setText("启动打磨")
            self.grinding_button.setStyleSheet("")
            self.grinding_status_indicator.setStyleSheet("background-color: #FF0000; border-radius: 10px;")
            self.grinding_progress_label.setText("打磨: 完成")
            self.speak_response("打磨已完成")

            # 清理任务ID
            self.current_grinding_task_id = None
        else:
            # 其他类型的任务
            print(f"未知任务 {task_id} 完成")

    def handle_task_failed(self, task_id, exception):
        """处理任务失败"""
        print(f"任务 {task_id} 失败: {exception}")

        if task_id == self.task_ids.get('camera'):
            # 显示错误消息或执行恢复操作
            pass
        elif task_id == self.task_ids.get('point_cloud'):
            # 点云数据获取任务失败
            self._handle_point_cloud_failed(exception)
            # 清理任务ID
            self.task_ids['point_cloud'] = None
        elif task_id == getattr(self, 'current_grinding_task_id', None):
            # 打磨任务失败
            print(f"打磨任务失败: {exception}")
            # 在主线程中安全更新UI
            self.grinding_button.setText("启动打磨")
            self.grinding_button.setStyleSheet("")
            self.grinding_status_indicator.setStyleSheet("background-color: #FF0000; border-radius: 10px;")
            self.grinding_progress_label.setText("打磨: 失败")
            self.speak_response("打磨任务失败")

            # 清理任务ID
            self.current_grinding_task_id = None
        else:
            # 其他类型的任务失败
            print(f"未知任务 {task_id} 失败")

    def start_camera_task(self):
        """启动摄像头任务"""
        if self.task_ids.get('camera') is not None:
            print("摄像头任务已在运行")
            return

        task_id = self.thread_pool.submit_task(self._camera_worker)
        self.task_ids['camera'] = task_id
        print(f"启动摄像头任务，ID: {task_id}")

    def _camera_worker(self):
        """摄像头工作线程"""
        try:
            # 摄像头处理逻辑
            while self.camera_active:
                frame = self.capture_frame()
                if frame is not None:
                    # 处理帧并返回结果
                    processed_frame = self.process_frame(frame)
                    return processed_frame
                time.sleep(0.03)
        except Exception as e:
            print(f"摄像头工作线程错误: {e}")
            raise

    def start_detection_task(self, image):
        """启动目标检测任务"""
        task_id = self.thread_pool.submit_task(self._detection_worker, image)
        self.task_ids['detection'] = task_id
        return task_id

    def _detection_worker(self, image):
        """目标检测工作线程"""
        try:
            # 使用YOLO模型进行目标检测
            boxes, scores, class_ids = self.yolo_model.detect(image)

            # 处理检测结果
            result = {
                'boxes': boxes,
                'scores': scores,
                'class_ids': class_ids,
                'timestamp': time.time()
            }

            return result
        except Exception as e:
            print(f"目标检测工作线程错误: {e}")
            raise

    def start_audio_processing_task(self, audio_data):
        """启动音频处理任务"""
        task_id = self.thread_pool.submit_task(self._audio_worker, audio_data)
        self.task_ids['audio'] = task_id
        return task_id

    def _audio_worker(self, audio_data):
        """音频处理工作线程"""
        try:
            # 语音识别处理
            text = self.asr_model.transcribe(audio_data)

            # 自然语言处理
            command = self.nlp_model.process(text)

            return {
                'text': text,
                'command': command,
                'timestamp': time.time()
            }
        except Exception as e:
            print(f"音频处理工作线程错误: {e}")
            raise

    def start_motion_task(self, target_position):
        """启动运动控制任务"""
        task_id = self.thread_pool.submit_task(self._motion_worker, target_position)
        self.task_ids['motion'] = task_id
        return task_id

    def _motion_worker(self, target_position):
        """运动控制工作线程"""
        try:
            # 运动规划和控制逻辑
            trajectory = self.planner.plan_trajectory(
                self.robot.get_current_position(),
                target_position
            )

            # 执行轨迹
            for point in trajectory:
                self.robot.move_to(point)
                time.sleep(0.1)  # 控制运动速度

            return {
                'success': True,
                'final_position': self.robot.get_current_position(),
                'timestamp': time.time()
            }
        except Exception as e:
            print(f"运动控制工作线程错误: {e}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': time.time()
            }

    def start_calibration_task(self):
        """启动标定任务"""
        task_id = self.thread_pool.submit_task(self._calibration_worker)
        self.task_ids['calibration'] = task_id
        return task_id

    def _calibration_worker(self):
        """标定工作线程"""
        try:
            # 执行标定流程
            calibration_data = self.calibrator.perform_calibration()

            # 保存标定结果
            self.calibrator.save_calibration(calibration_data)

            return {
                'success': True,
                'data': calibration_data,
                'timestamp': time.time()
            }
        except Exception as e:
            print(f"标定工作线程错误: {e}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': time.time()
            }

    def stop_all_tasks(self):
        """停止所有任务"""
        # 停止特定类型的任务
        self.camera_active = False

        # 等待所有任务完成
        self.thread_pool.shutdown(wait=True)

        # 重置任务ID
        for key in self.task_ids:
            self.task_ids[key] = None

    def closeEvent(self, event):
        """关闭应用程序时清理资源"""
        self.stop_all_tasks()
        super().closeEvent(event)