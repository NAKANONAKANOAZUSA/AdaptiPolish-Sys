import time
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QGroupBox, QMessageBox,
    QInputDialog
)
from PyQt5.QtCore import Qt, QTimer
class ManualControlDialog(QDialog):
    """手动控制对话框"""

    def __init__(self, connection, parent=None):
        super().__init__(parent)
        self.setWindowTitle("手动控制模式")
        self.setGeometry(200, 200, 500, 600)
        self.setStyleSheet("""
            QDialog {
                background-color: #2D2D30;
                color: #FFFFFF;
                font-family: Segoe UI;
            }
            QGroupBox {
                border: 1px solid #007ACC;
                border-radius: 6px;
                margin-top: 10px;
                padding-top: 15px;
                font-weight: bold;
            }
            QLabel {
                color: #CCCCCC;
            }
            QPushButton {
                background-color: #007ACC;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 6px 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1C97EA;
            }
            QPushButton:disabled {
                background-color: #505050;
                color: #A0A0A0;
            }
        """)

        self.connection = connection
        self.init_ui()
        QTimer.singleShot(100, self.update_all_status)

        # 添加自动更新状态计时器
        self.status_timer = QTimer(self)
        self.status_timer.timeout.connect(self.update_joint_status)
        self.status_timer.start(5000)

    def init_ui(self):
        """初始化UI"""
        main_layout = QVBoxLayout()
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(15, 15, 15, 15)

        # 关节控制区域
        joint_group = QGroupBox("关节控制")
        joint_layout = QGridLayout()
        joint_layout.setHorizontalSpacing(10)
        joint_layout.setVerticalSpacing(10)

        # 创建关节标签和按钮
        self.joint_btns = []
        for i in range(1, 7):
            label = QLabel(f"关节 {i}:")
            label.setAlignment(Qt.AlignCenter)

            enable_btn = QPushButton("使能")
            enable_btn.setFixedHeight(30)

            release_btn = QPushButton("放松")
            release_btn.setFixedHeight(30)

            status_label = QLabel("状态: 未知")

            self.joint_btns.append({
                'enable_btn': enable_btn,
                'release_btn': release_btn,
                'status_label': status_label
            })

            enable_btn.clicked.connect(lambda _, idx=i: self.focus_joint(idx))
            release_btn.clicked.connect(lambda _, idx=i: self.release_joint(idx))

            row = i - 1
            joint_layout.addWidget(label, row, 0)
            joint_layout.addWidget(enable_btn, row, 1)
            joint_layout.addWidget(release_btn, row, 2)
            joint_layout.addWidget(status_label, row, 3)

        # 添加全部关节控制按钮
        all_joints_layout = QHBoxLayout()
        enable_all_btn = QPushButton("使能所有关节")
        enable_all_btn.setFixedHeight(35)
        release_2_6_btn = QPushButton("放松1-6关节")
        release_2_6_btn.setFixedHeight(35)

        enable_all_btn.clicked.connect(self.power_on_all_joints)
        release_2_6_btn.clicked.connect(lambda: self.release_joints([1, 2, 3, 4, 5, 6]))

        all_joints_layout.addWidget(enable_all_btn)
        all_joints_layout.addWidget(release_2_6_btn)

        joint_layout.addLayout(all_joints_layout, 6, 0, 1, 4)

        joint_group.setLayout(joint_layout)

        # 自由移动模式控制
        free_group = QGroupBox("自由移动模式")
        free_layout = QVBoxLayout()
        free_layout.setSpacing(10)

        self.free_mode_switch = QPushButton("点击按钮获取状态")
        self.free_mode_switch.setFixedHeight(40)
        self.free_mode_switch.clicked.connect(self.toggle_free_mode)

        check_status_btn = QPushButton("检查自由移动状态")
        check_status_btn.setFixedHeight(35)
        check_status_btn.clicked.connect(self.update_free_mode_status)

        free_layout.addWidget(self.free_mode_switch)
        free_layout.addWidget(check_status_btn)
        free_group.setLayout(free_layout)

        # 保存点功能
        save_group = QGroupBox("保存点")
        save_layout = QHBoxLayout()
        save_point_btn = QPushButton("保存当前点位")
        save_point_btn.setFixedHeight(40)
        save_point_btn.clicked.connect(self.save_current_point)
        save_layout.addWidget(save_point_btn)
        save_group.setLayout(save_layout)

        # 底部按钮
        bottom_layout = QHBoxLayout()
        exit_btn = QPushButton("退出手动模式")
        exit_btn.setFixedHeight(40)
        exit_btn.setStyleSheet("background-color: #FF4500;")
        exit_btn.clicked.connect(self.exit_manual_mode)

        refresh_btn = QPushButton("刷新状态")
        refresh_btn.setFixedHeight(35)
        refresh_btn.clicked.connect(self.update_all_status)

        bottom_layout.addWidget(refresh_btn)
        bottom_layout.addStretch()
        bottom_layout.addWidget(exit_btn)

        # 添加组件到主布局
        main_layout.addWidget(joint_group)
        main_layout.addWidget(free_group)
        main_layout.addWidget(save_group)
        main_layout.addLayout(bottom_layout)

        self.setLayout(main_layout)
        self.update_all_status()

    def update_all_status(self):
        """更新所有状态显示"""
        self.update_joint_status()
        self.update_free_mode_status()

    def update_joint_status(self):
        """更新关节状态显示"""
        if not self.connection or not self.connection.is_connected():
            for i in range(1, 7):
                self.joint_btns[i - 1]['status_label'].setText("状态: 未连接")
            return

        try:
            mc = self.connection.get_robot()
            voltages = mc.get_servo_voltages()
            temps = mc.get_servo_temps()

            for i in range(1, 7):
                enabled = mc.is_servo_enable(i) == 1

                # 获取电压和温度
                voltage = "N/A"
                temp = "N/A"
                if voltages and len(voltages) >= i:
                    voltage = voltages[i - 1]
                if temps and len(temps) >= i:
                    temp = temps[i - 1]

                status_text = f"电压: {voltage}V, 温度: {temp}°C"
                self.joint_btns[i - 1]['status_label'].setText(status_text)

                # 更新按钮状态
                self.joint_btns[i - 1]['enable_btn'].setEnabled(enabled)
                self.joint_btns[i - 1]['release_btn'].setEnabled(enabled)

        except Exception as e:
            print(f"更新关节状态出错: {str(e)}")
            # 设置错误状态
            for i in range(1, 7):
                self.joint_btns[i - 1]['status_label'].setText("状态: 获取失败")

    def update_free_mode_status(self):
        """更新自由移动模式状态"""
        if not self.connection or not self.connection.is_connected():
            QMessageBox.warning(self, "未连接", "机械臂未连接")
            self.free_mode_switch.setText("状态: 未连接")
            self.free_mode_switch.setStyleSheet("background-color: #505050;")
            return

        try:
            mc = self.connection.get_robot()
            status = mc.is_free_mode()
            if status == 1:
                self.free_mode_switch.setText("自由移动模式: 已启用 (点击关闭)")
                self.free_mode_switch.setStyleSheet("background-color: #00FF00; color: black;")
            elif status == 0:
                self.free_mode_switch.setText("自由移动模式: 已禁用 (点击开启)")
                self.free_mode_switch.setStyleSheet("background-color: #FF0000; color: white;")
            else:
                self.free_mode_switch.setText("状态未知")
                self.free_mode_switch.setStyleSheet("background-color: #505050;")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"获取自由移动状态失败: {str(e)}")
            self.free_mode_switch.setText("状态: 错误")
            self.free_mode_switch.setStyleSheet("background-color: #FF4500;")

    def toggle_free_mode(self):
        """切换自由移动模式"""
        if not self.connection or not self.connection.is_connected():
            QMessageBox.warning(self, "未连接", "机械臂未连接")
            return

        try:
            mc = self.connection.get_robot()
            current_status = mc.is_free_mode()
            if current_status == 1:
                result = mc.set_free_mode(0)
                if result == 1:
                    self.update_free_mode_status()
                    QMessageBox.information(self, "成功", "自由移动模式已禁用")
                else:
                    QMessageBox.warning(self, "失败", f"禁用自由移动模式失败 (返回值: {result})")
            else:
                result = mc.set_free_mode(1)
                if result == 1:
                    self.update_free_mode_status()
                    QMessageBox.information(self, "成功", "自由移动模式已启用")
                else:
                    QMessageBox.warning(self, "失败", f"启用自由移动模式失败 (返回值: {result})")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"设置自由移动模式失败: {str(e)}")

    def focus_joint(self, joint_id):
        """上电指定关节"""
        if not self.connection or not self.connection.is_connected():
            QMessageBox.warning(self, "未连接", "机械臂未连接")
            return

        try:
            mc = self.connection.get_robot()
            if 1 <= joint_id <= 6:
                result = mc.focus_servo(joint_id)
                if result == 1:
                    self.update_joint_status()
                    QMessageBox.information(self, "成功", f"关节 {joint_id} 已上电")
                else:
                    QMessageBox.warning(self, "失败", f"上电关节 {joint_id} 失败 (返回值: {result})")
            else:
                QMessageBox.warning(self, "错误", "无效的关节ID (1-6)")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"上电关节时发生错误: {str(e)}")

    def release_joint(self, joint_id):
        """放松指定关节"""
        if not self.connection or not self.connection.is_connected():
            QMessageBox.warning(self, "未连接", "机械臂未连接")
            return

        try:
            mc = self.connection.get_robot()
            if 1 <= joint_id <= 6:
                result = mc.release_servo(joint_id)
                self.update_joint_status()
                QMessageBox.information(self, "成功", f"关节 {joint_id} 已放松")
            else:
                QMessageBox.warning(self, "错误", "无效的关节ID (1-6)")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"放松关节时发生错误: {str(e)}")

    def power_on_all_joints(self):
        """上电所有关节"""
        if not self.connection or not self.connection.is_connected():
            QMessageBox.warning(self, "未连接", "机械臂未连接")
            return

        try:
            mc = self.connection.get_robot()
            result = mc.focus_all_servos()
            if result == 1:
                self.update_joint_status()
                QMessageBox.information(self, "成功", "所有关节已上电")
            else:
                QMessageBox.warning(self, "失败", f"上电所有关节失败 (返回值: {result})")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"上电所有关节时发生错误: {str(e)}")

    def release_joints(self, joint_ids):
        """放松指定关节列表"""
        results = []

        for joint_id in joint_ids:
            try:
                mc = self.connection.get_robot()
                if 1 <= joint_id <= 6:
                    result = mc.release_servo(joint_id)
                    if result == 1:
                        results.append(True)
                    else:
                        results.append(False)
                        print(f"放松关节 {joint_id} 失败 (返回值: {result})")
                else:
                    print(f"无效的关节ID: {joint_id}")
            except Exception as e:
                print(f"放松关节 {joint_id} 时出错: {str(e)}")
                results.append(False)

        # 更新关节状态
        self.update_joint_status()

        # 检查是否所有操作都成功
        if all(results):
            QMessageBox.information(self, "操作警告", "部分关节放松失败，请查看日志")
        else:
            QMessageBox.warning(self, "操作完成", "已放松1-6关节")

    def save_current_point(self):
        """保存当前点位"""
        name, ok = QInputDialog.getText(self, "保存点位", "输入点位名称:")
        if ok and name:
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

                    QMessageBox.information(self, "保存成功", f"点位 '{name}' 已保存")
                    # 查询当前世界坐标系下的坐标
                    world_coords = mc.get_coords()
                    print("Current world coordinates:", world_coords)
                    if self.parent():
                        self.parent().save_teach_point_object(point)
                else:
                    QMessageBox.warning(self, "获取位置失败", "无法获取机器人当前位置")
            except Exception as e:
                QMessageBox.critical(self, "保存错误", f"保存示教点时出错: {str(e)}")
        else:
            QMessageBox.warning(self, "输入错误", "请输入有效的点位名称")

    def exit_manual_mode(self):
        """退出手动模式"""
        if self.connection and self.connection.is_connected():
            try:
                # 尝试禁用自由移动模式
                mc = self.connection.get_robot()
                if mc.is_free_mode() == 1:
                    mc.set_free_mode(0)
                    print("自由移动模式已禁用")

                # 尝试上电所有关节
                self.power_on_all_joints()
            except Exception as e:
                print(f"退出手动模式时出错: {str(e)}")

        self.accept()