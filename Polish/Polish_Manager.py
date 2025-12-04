import json
import threading
import time


class GrindingController:
    """打磨控制器 - 负责所有打磨相关的功能"""

    def __init__(self, config, robot_connection, detection_thread, motor_controller):
        self.config = config
        self.robot_connection = robot_connection
        self.detection_thread = detection_thread
        self.motor_controller = motor_controller

        # 打磨参数
        self.grinding_loops = 1
        self.grinding_x_step = 0.0
        self.grinding_y_step = 0.0
        self.grinding_z_step = 0.0
        self.path_scale_factor_X = 1.0
        self.path_scale_factor_Y = 1.0
        self.grinding_current_loop = 0
        self.base_distance = 245
        self.base_scale_x = 0.67
        self.base_scale_y = 0.55
        self.current_distance = 245

        # 历史路径管理
        self.history_paths = []
        self.current_history_path = None
        self.use_history_path = False
        self.coordinate_rotation = 270

        # 用户偏移量
        self.user_offset_x = 0
        self.user_offset_y = 0
        self.user_offset_z = 0

        # 状态标志
        self.is_grinding = False
        self.grinding_thread = None

    def toggle_grinding(self, use_history_path=False, current_history_path=None):
        """切换打磨状态"""
        self.use_history_path = use_history_path
        self.current_history_path = current_history_path

        if not self.is_grinding:
            return self.start_grinding()
        else:
            return self.stop_grinding()

    def start_grinding(self):
        """启动打磨"""
        if not self.robot_connection or not self.robot_connection.is_connected():
            return False, "未连接到机器人"

        self.is_grinding = True

        # 使用历史路径时跳过检测
        if self.use_history_path and self.current_history_path:
            print("使用历史路径，跳过目标检测")
            contour_points = self.process_history_path()
            if not contour_points:
                return False, "无法处理历史路径"
        else:
            # 获取轮廓点
            contour_points = []
            if (hasattr(self.detection_thread, 'detection_system') and
                    hasattr(self.detection_thread.detection_system, 'fixed_contour')):
                contour_points = self.detection_thread.detection_system.fixed_contour

            if not contour_points:
                return False, "未找到轮廓点"

        # 创建并启动打磨线程
        self.grinding_thread = threading.Thread(
            target=self._run_grinding_procedure,
            args=(contour_points, self.user_offset_x, self.user_offset_y, self.user_offset_z)
        )
        self.grinding_thread.daemon = True
        self.grinding_thread.start()

        return True, "打磨程序已启动"

    def stop_grinding(self):
        """停止打磨"""
        self.is_grinding = False

        # 停止机械臂
        if self.robot_connection:
            try:
                self.robot_connection.get_robot().stop()
            except Exception as e:
                print(f"停止机械臂时出错: {e}")

        # 停止电机
        if self.motor_controller:
            try:
                self.motor_controller.emergency_stop()
            except Exception as e:
                print(f"停止电机时出错: {e}")

        return True, "打磨程序已停止"

    def _run_grinding_procedure(self, contour_points, user_offset_x, user_offset_y, user_offset_z=260):
        """执行打磨流程的核心方法"""
        if not self.robot_connection or not self.robot_connection.is_connected():
            print("未连接到机器人，无法执行打磨")
            return

        robot = self.robot_connection.get_robot()
        original_mode = None
        original_ref_frame = None
        motor_started = False

        try:
            # 启动电机
            if self.motor_controller:
                try:
                    speed = getattr(self, 'motor_max_speed', 400)
                    print(f"启动打磨电机，转速: {speed} RPM")
                    self.motor_controller.stop()
                    self.motor_controller.set_speed(speed)
                    self.motor_controller.forward()
                    motor_started = True
                    time.sleep(0.5)
                except Exception as motor_err:
                    print(f"启动电机失败: {motor_err}")

            # 保存原始运动模式和参考坐标系
            original_mode = robot.get_fresh_mode()
            original_ref_frame = robot.get_reference_frame()

            # 配置优化运动模式
            robot.set_fresh_mode(1)
            robot.set_movement_type(1)
            robot.set_end_type(0)
            robot.set_reference_frame(0)

            # 固定姿态参数
            fixed_rx, fixed_ry, fixed_rz = 179.87, -3.78, -179.75

            # 轮廓点检查
            if not contour_points or len(contour_points) < 3:
                print("轮廓点不足，无法执行打磨")
                return

            detection_system = self.detection_thread.detection_system

            # 保存原始偏移量
            original_offset_x = user_offset_x
            original_offset_y = user_offset_y
            original_offset_z = user_offset_z

            # 重置当前循环计数
            self.grinding_current_loop = 0

            # 速度参数
            APPROACH_SPEED = 40
            GRIND_SPEED = 5
            JOINT_SPEED = 5

            # 循环执行打磨
            for loop in range(self.grinding_loops):
                if not self.is_grinding:
                    break

                self.grinding_current_loop = loop + 1
                print(f"====== 开始打磨循环 {self.grinding_current_loop}/{self.grinding_loops} ======")

                # 计算当前循环的偏移量
                current_x_offset = original_offset_x + (self.grinding_x_step * loop)
                current_y_offset = original_offset_y + (self.grinding_y_step * loop)
                current_z_offset = original_offset_z + (self.grinding_z_step * loop)

                # 应用缩放比例
                scaled_points = []
                for point in contour_points:
                    scaled_x = point[0] * self.path_scale_factor_Y
                    scaled_y = point[1] * self.path_scale_factor_X
                    scaled_points.append([scaled_x, scaled_y])

                # 转换为世界坐标
                world_contour = []
                for point in scaled_points:
                    pixel_x, pixel_y = point
                    world_x, world_y, success = detection_system.pixel_to_world_coords(pixel_x, pixel_y)
                    if not success:
                        continue

                    world_contour.append([
                        world_x + current_x_offset,
                        world_y + current_y_offset,
                        current_z_offset
                    ])

                # 运动序列优化
                safe_angles = [0.79, 53.43, -129.81, -8.96, 2.02, 90.17]
                start_point = world_contour[0]

                # 移动到起始点上方
                approach_height = current_z_offset + 10
                approach_position = [start_point[0], start_point[1], approach_height, fixed_rx, fixed_ry, fixed_rz]
                robot.sync_send_coords(approach_position, APPROACH_SPEED, mode=1, timeout=8)

                # 下降到打磨高度
                grind_position = [start_point[0], start_point[1], current_z_offset, fixed_rx, fixed_ry, fixed_rz]
                robot.sync_send_coords(grind_position, GRIND_SPEED, mode=1, timeout=8)

                # 创建完整的运动轨迹
                path_points = []
                for point in world_contour:
                    if not self.is_grinding:
                        break
                    path_points.append([point[0], point[1], point[2], fixed_rx, fixed_ry, fixed_rz])

                # 获取当前角度
                current_angles = robot.get_angles() or safe_angles
                lock_j6 = current_angles[5] if len(current_angles) >= 6 else 90.17

                # 创建关节角度路径点
                joint_path_points = []
                for i, coords in enumerate(path_points):
                    try:
                        angles = robot.solve_inv_kinematics(coords, current_angles)
                    except Exception as e:
                        angles = joint_path_points[-1] if joint_path_points else current_angles

                    angles[5] = lock_j6
                    angles[3] = angles[3] + 13.5
                    joint_path_points.append(angles)
                    current_angles = angles

                # 切换到关节运动模式
                robot.set_movement_type(0)

                # 执行关节运动
                for angles in joint_path_points:
                    if not self.is_grinding:
                        break
                    robot.sync_send_angles(angles, JOINT_SPEED, timeout=0.05)
                    time.sleep(0.05)

                # 抬升工具头
                if path_points:
                    lift_position = [
                        path_points[-1][0], path_points[-1][1], approach_height, fixed_rx, fixed_ry, fixed_rz
                    ]
                    robot.sync_send_coords(lift_position, APPROACH_SPEED, mode=1, timeout=5)

                print(f"====== 完成打磨循环 {self.grinding_current_loop}/{self.grinding_loops} ======")

                if loop < self.grinding_loops - 1:
                    time.sleep(1)

            # 返回安全位置
            robot.sync_send_angles(safe_angles, APPROACH_SPEED, timeout=8)

        except Exception as e:
            print(f"打磨过程中发生错误: {str(e)}")
            import traceback
            traceback.print_exc()
        finally:
            # 恢复原始设置
            if original_mode is not None:
                try:
                    robot.set_fresh_mode(original_mode)
                    robot.set_movement_type(0)
                except Exception as e:
                    print(f"恢复原始模式错误: {e}")

            # 停止电机
            if motor_started and self.motor_controller:
                try:
                    self.motor_controller.emergency_stop()
                except Exception as e:
                    print(f"停止电机时出错: {e}")

            self.is_grinding = False

    def apply_grinding_params(self, loops, x_step, y_step, z_step, scale_x, scale_y):
        """应用打磨参数"""
        try:
            self.grinding_loops = max(1, min(100, int(loops)))
            self.grinding_x_step = max(-10.0, min(10.0, float(x_step)))
            self.grinding_y_step = max(-10.0, min(10.0, float(y_step)))
            self.grinding_z_step = max(-10.0, min(10.0, float(z_step)))
            self.path_scale_factor_X = max(0.1, min(10.0, float(scale_x)))
            self.path_scale_factor_Y = max(0.1, min(10.0, float(scale_y)))
            return True
        except ValueError:
            return False

    def update_distance_from_sensor(self):
        """从测距传感器获取最新距离"""
        try:
            # 暂时返回固定值作为示例
            distance = 245
            self.current_distance = distance
            self.update_scaling_factors(distance)
            return True
        except Exception as e:
            print(f"获取距离数据失败: {str(e)}")
            return False

    def update_scaling_factors(self, distance):
        """根据距离更新缩放比例"""
        try:
            if distance < 10:
                distance = self.base_distance

            self.current_distance = distance
            new_scale_x = round((self.base_distance * self.base_scale_x) / distance, 4)
            new_scale_y = round((self.base_distance * self.base_scale_y) / distance, 4)

            self.path_scale_factor_X = new_scale_x
            self.path_scale_factor_Y = new_scale_y

            return True
        except Exception as e:
            print(f"更新缩放比例失败: {str(e)}")
            return False

    def check_path_points(self, contour_points):
        """检查生成的路径点"""
        if not contour_points:
            print("没有生成路径点")
            return

        print(f"=== 路径点检查 ===")
        print(f"原始点数: {len(contour_points)}")

        x_points = [p[0] for p in contour_points]
        y_points = [p[1] for p in contour_points]
        min_x, max_x = min(x_points), max(x_points)
        min_y, max_y = min(y_points), max(y_points)

        print(f"X范围: {min_x:.2f} - {max_x:.2f}, 宽度: {max_x - min_x:.2f}")
        print(f"Y范围: {min_y:.2f} - {max_y:.2f}, 高度: {max_y - min_y:.2f}")

    def process_history_path(self):
        """处理历史路径用于打磨"""
        if not self.current_history_path:
            return None

        if not self.ensure_detection_system():
            return None

        if not self.current_history_path.get('points'):
            return None

        rotation = self.current_history_path.get('rotation', 90)
        world_coords = self.convert_path_to_world_coords(
            self.current_history_path['points'],
            self.current_history_path.get('original_size', (640, 480))
        )

        if world_coords is None:
            return None

        self.current_history_path['world_coords'] = world_coords
        self.save_history_paths()

        return world_coords

    def convert_path_to_world_coords(self, pixel_points, original_size):
        """使用检测系统转换路径到世界坐标"""
        if not hasattr(self.detection_thread, 'detection_system'):
            return None

        detection_system = self.detection_thread.detection_system
        world_coords = []

        for point in pixel_points:
            if len(point) < 2:
                continue

            scaled_x = point[0] * self.path_scale_factor_Y
            scaled_y = point[1] * self.path_scale_factor_X

            world_x, world_y, success = detection_system.pixel_to_world_coords(
                scaled_x, scaled_y, rotation=self.coordinate_rotation
            )
            if not success:
                continue

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

    def ensure_detection_system(self):
        """确保检测系统可用"""
        if hasattr(self.detection_thread, 'detection_system'):
            return True

        try:
            # 这里需要根据实际情况初始化检测系统
            return True
        except Exception as e:
            print(f"初始化检测系统失败: {str(e)}")
            return False

    def set_user_offsets(self, x, y, z):
        """设置用户偏移量"""
        self.user_offset_x = x
        self.user_offset_y = y
        self.user_offset_z = z

    def get_grinding_status(self):
        """获取打磨状态"""
        return {
            'is_grinding': self.is_grinding,
            'current_loop': self.grinding_current_loop,
            'total_loops': self.grinding_loops,
            'progress': (self.grinding_current_loop / self.grinding_loops * 100) if self.grinding_loops > 0 else 0
        }