from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QGridLayout,
                             QGroupBox, QLabel, QPushButton, QMessageBox, QSlider)
from PyQt5.QtCore import Qt
from hardware.motor import MotorController
class MotorControlDialog(QDialog):
    """电机控制对话框"""

    def __init__(self,parent=None):
        super().__init__(parent)
        self.setWindowTitle("电机控制")
        self.setGeometry(300, 300, 400, 400)
        self.motor = None
        self.init_ui()

    def init_ui(self):
        """初始化UI"""
        main_layout = QVBoxLayout()
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(15, 15, 15, 15)

        # 状态指示区域
        status_group = QGroupBox("电机状态")
        status_layout = QVBoxLayout()

        self.status_label = QLabel("未连接")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("font-size: 14px; font-weight: bold;")

        status_layout.addWidget(self.status_label)
        status_group.setLayout(status_layout)

        # 基本控制区域
        control_group = QGroupBox("基本控制")
        control_layout = QGridLayout()

        self.start_button = QPushButton("启动")
        self.start_button.clicked.connect(self.start_motor)

        self.stop_button = QPushButton("停止")
        self.stop_button.clicked.connect(self.stop_motor)

        self.forward_button = QPushButton("正转")
        self.forward_button.clicked.connect(self.forward_motor)

        self.reverse_button = QPushButton("反转")
        self.reverse_button.clicked.connect(self.reverse_motor)

        self.emergency_button = QPushButton("急停")
        self.emergency_button.setStyleSheet("background-color: #FF4500;")
        self.emergency_button.clicked.connect(self.emergency_stop)

        control_layout.addWidget(self.start_button, 0, 0)
        control_layout.addWidget(self.stop_button, 0, 1)
        control_layout.addWidget(self.forward_button, 1, 0)
        control_layout.addWidget(self.reverse_button, 1, 1)
        control_layout.addWidget(self.emergency_button, 2, 0, 1, 2)

        control_group.setLayout(control_layout)

        # 速度控制区域
        speed_group = QGroupBox("速度控制")
        speed_layout = QVBoxLayout()

        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setRange(0, 500)
        self.speed_slider.setValue(300)
        self.speed_slider.valueChanged.connect(self.update_speed_label)

        self.speed_label = QLabel("速度: 300")
        self.speed_label.setAlignment(Qt.AlignCenter)

        self.set_speed_button = QPushButton("设置速度")
        self.set_speed_button.clicked.connect(self.set_motor_speed)

        speed_layout.addWidget(self.speed_slider)
        speed_layout.addWidget(self.speed_label)
        speed_layout.addWidget(self.set_speed_button)
        speed_group.setLayout(speed_layout)

        # 连接按钮
        self.connect_button = QPushButton("连接电机")
        self.connect_button.clicked.connect(self.toggle_connection)
        self.connect_button.setFixedHeight(40)

        # 添加组件到主布局
        main_layout.addWidget(status_group)
        main_layout.addWidget(control_group)
        main_layout.addWidget(speed_group)
        main_layout.addWidget(self.connect_button)

        self.setLayout(main_layout)
        self.set_controls_enabled(False)

    def toggle_connection(self):
        """切换电机连接状态"""
        if self.connect_button.text() == "连接电机":
            try:
                self.motor = MotorController(port='/dev/ttyCH343USB0')
                self.status_label.setText("已连接")
                self.status_label.setStyleSheet("color: green; font-size: 14px; font-weight: bold;")
                self.connect_button.setText("断开连接")
                self.set_controls_enabled(True)
                QMessageBox.information(self, "连接成功", "已成功连接到电机控制器")
            except Exception as e:
                QMessageBox.critical(self, "连接失败", f"无法连接到电机控制器: {str(e)}")
                self.status_label.setText("连接失败")
                self.status_label.setStyleSheet("color: red; font-size: 14px; font-weight: bold;")
        else:
            if self.motor:
                try:
                    self.motor.close()
                except:
                    pass
                self.motor = None
            self.status_label.setText("未连接")
            self.status_label.setStyleSheet("color: gray; font-size: 14px; font-weight: bold;")
            self.connect_button.setText("连接电机")
            self.set_controls_enabled(False)
            QMessageBox.information(self, "断开连接", "已断开与电机控制器的连接")

    def set_controls_enabled(self, enabled):
        """设置控制按钮的启用状态"""
        self.start_button.setEnabled(enabled)
        self.stop_button.setEnabled(enabled)
        self.forward_button.setEnabled(enabled)
        self.reverse_button.setEnabled(enabled)
        self.emergency_button.setEnabled(enabled)
        self.speed_slider.setEnabled(enabled)
        self.set_speed_button.setEnabled(enabled)

    def update_speed_label(self, value):
        """更新速度标签"""
        self.speed_label.setText(f"速度: {value}")

    def start_motor(self):
        """启动电机"""
        if self.motor:
            try:
                self.motor.start()
                self.status_label.setText("电机已启动")
                QMessageBox.information(self, "成功", "电机已启动")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"启动电机失败: {str(e)}")

    def stop_motor(self):
        """停止电机"""
        if self.motor:
            try:
                self.motor.stop()
                self.status_label.setText("电机已停止")
                QMessageBox.information(self, "成功", "电机已停止")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"停止电机失败: {str(e)}")

    def forward_motor(self):
        """正转电机"""
        if self.motor:
            try:
                self.motor.forward()
                self.status_label.setText("电机正转中")
                QMessageBox.information(self, "成功", "电机正在正转")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"设置正转失败: {str(e)}")

    def reverse_motor(self):
        """反转电机"""
        if self.motor:
            try:
                self.motor.reverse()
                self.status_label.setText("电机反转中")
                QMessageBox.information(self, "成功", "电机正在反转")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"设置反转失败: {str(e)}")

    def emergency_stop(self):
        """急停电机"""
        if self.motor:
            try:
                self.motor.emergency_stop()
                self.status_label.setText("电机已急停")
                QMessageBox.information(self, "成功", "电机已执行急停")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"执行急停失败: {str(e)}")

    def set_motor_speed(self):
        """设置电机转速"""
        if self.motor:
            speed = self.speed_slider.value()
            try:
                self.motor.set_speed(speed)
                self.status_label.setText(f"转速已设置为: {speed}")
                QMessageBox.information(self, "成功", f"电机转速已设置为: {speed}")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"设置转速失败: {str(e)}")

    def closeEvent(self, event):
        """关闭窗口时断开连接"""
        if self.motor:
            try:
                self.motor.close()
            except:
                pass
        event.accept()
