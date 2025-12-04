from pymycobot import MyCobot280
import time
import math
import numpy as np


class MyCobotGrindingController:
    """
    MyCobot280 打磨路径控制类
    智能选择主导关节，根据路径方向优化运动策略
    """

    def __init__(self, mycobot_instance, port='/dev/ttyAMA0', baudrate=1000000):
        """
        初始化机械臂控制器

        Args:
            mycobot_instance: MyCobot280实例
            port: 串口地址
            baudrate: 波特率
        """
        self.mc = mycobot_instance

        # 各关节角度限制
        self.JOINT_LIMITS = {
            1: (-55, 84.9),  # 一轴
            2: (-60, 60),  # 二轴
            3: (-120, 0),  # 三轴
            4: (-90, 90),  # 四轴
            5: (-90, 90),  # 五轴
            6: (-90, 90)  # 六轴
        }

        # 默认运动速度
        self.default_speed = 20

        print(f"MyCobot280 控制器初始化完成，端口: {port}, 波特率: {baudrate}")

    def is_joint_safe(self, joint_id, angle):
        """检查指定关节角度是否在安全范围内"""
        min_angle, max_angle = self.JOINT_LIMITS[joint_id]
        return min_angle <= angle <= max_angle

    def adjust_joint_angle(self, joint_id, angle):
        """调整指定关节角度到安全范围内"""
        min_angle, max_angle = self.JOINT_LIMITS[joint_id]
        if angle < min_angle:
            return min_angle
        elif angle > max_angle:
            return max_angle
        else:
            return angle

    def analyze_path_direction(self, start_coords, end_coords):
        """
        分析路径方向，确定最适合的主导关节

        Args:
            start_coords: 起点坐标
            end_coords: 终点坐标

        Returns:
            dict: 包含主导关节建议和分析结果
        """
        # 计算位置变化量
        dx = end_coords[0] - start_coords[0]
        dy = end_coords[1] - start_coords[1]
        dz = end_coords[2] - start_coords[2]

        # 计算各方向变化量的绝对值
        abs_dx = abs(dx)
        abs_dy = abs(dy)
        abs_dz = abs(dz)

        # 判断主要运动方向
        max_movement = max(abs_dx, abs_dy, abs_dz)

        result = {
            'dx': dx, 'dy': dy, 'dz': dz,
            'abs_dx': abs_dx, 'abs_dy': abs_dy, 'abs_dz': abs_dz,
            'primary_direction': None,
            'recommended_joints': []
        }

        # 根据主要运动方向推荐主导关节
        if max_movement == abs_dx:  # X方向为主
            result['primary_direction'] = 'x'
            # X方向运动主要由关节2和关节3控制
            result['recommended_joints'] = [2, 3, 1]  # 优先级：关节2 > 关节3 > 关节1
            print(f"路径分析: X方向运动为主 (Δx={dx:.1f}mm)，推荐使用关节2和关节3作为主导")

        elif max_movement == abs_dy:  # Y方向为主
            result['primary_direction'] = 'y'
            # Y方向运动主要由关节1控制
            result['recommended_joints'] = [1, 2, 3]  # 优先级：关节1 > 关节2 > 关节3
            print(f"路径分析: Y方向运动为主 (Δy={dy:.1f}mm)，推荐使用关节1作为主导")

        else:  # Z方向为主
            result['primary_direction'] = 'z'
            # Z方向运动主要由关节2和关节3控制
            result['recommended_joints'] = [2, 3, 1]  # 优先级：关节2 > 关节3 > 关节1
            print(f"路径分析: Z方向运动为主 (Δz={dz:.1f}mm)，推荐使用关节2和关节3作为主导")

        return result

    def generate_grinding_path(self, center_coords, path_length=20.0, path_direction='x', tool_orientation=None):
        """
        根据中心点坐标生成打磨路径的起点和终点

        参数:
        - center_coords: 打磨路径的中心点坐标 [x, y, z, rx, ry, rz]
        - path_length: 打磨路径的长度（毫米），默认20mm
        - path_direction: 打磨路径的方向，可选 'x', 'y', 'z' 或自定义向量 [dx, dy, dz]
        - tool_orientation: 工具姿态，如果为None则使用中心点的姿态

        返回:
        - start_coords: 起点坐标
        - end_coords: 终点坐标
        """

        # 提取位置和姿态
        center_pos = np.array(center_coords[:3])
        if tool_orientation is None:
            tool_orientation = center_coords[3:]
        else:
            tool_orientation = np.array(tool_orientation)

        # 计算路径方向向量
        if path_direction == 'x':
            direction = np.array([1, 0, 0])
        elif path_direction == 'y':
            direction = np.array([0, 1, 0])
        elif path_direction == 'z':
            direction = np.array([0, 0, 1])
        elif isinstance(path_direction, (list, tuple, np.ndarray)) and len(path_direction) >= 3:
            direction = np.array(path_direction[:3])
            # 归一化方向向量
            direction_norm = np.linalg.norm(direction)
            if direction_norm > 0:
                direction = direction / direction_norm
            else:
                direction = np.array([1, 0, 0])  # 默认方向
        else:
            direction = np.array([1, 0, 0])  # 默认X方向

        # 计算起点和终点
        half_length = path_length / 2.0
        start_pos = center_pos - direction * half_length
        end_pos = center_pos + direction * half_length

        # 组合坐标和姿态
        start_coords = list(start_pos) + list(tool_orientation)
        end_coords = list(end_pos) + list(tool_orientation)

        print(f"生成的打磨路径:")
        print(f"中心点: {center_coords}")
        print(f"路径方向: {direction}")
        print(f"路径长度: {path_length}mm")
        print(f"起点: {start_coords}")
        print(f"终点: {end_coords}")

        return start_coords, end_coords

    def find_optimal_movement_solution(self, start_coords, end_coords, preferred_joints=None, max_attempts=30):
        """
        寻找最优运动解决方案，考虑路径方向选择主导关节

        Args:
            start_coords: 起点坐标 [x, y, z, rx, ry, rz]
            end_coords: 终点坐标 [x, y, z, rx, ry, rz]
            preferred_joints: 优先使用的主导关节列表，如[1, 2, 3]
            max_attempts: 最大尝试次数

        Returns:
            tuple: (angles_start, angles_end, dominant_joint, dominant_movement, other_joints_movement, score) 或 None
        """
        if preferred_joints is None:
            preferred_joints = [1, 2, 3]  # 默认优先使用前三个关节

        current_angles = self.mc.get_angles()
        best_solution = None
        best_score = -float('inf')

        print("寻找最优运动解决方案...")

        for attempt in range(max_attempts):
            try:
                # 使用不同的初始角度尝试
                if attempt == 0:
                    init_angles = current_angles
                else:
                    # 生成随机的初始角度，确保各关节在安全范围内
                    init_angles = []
                    for joint_id in range(1, 7):
                        min_angle, max_angle = self.JOINT_LIMITS[joint_id]
                        init_angles.append(np.random.uniform(min_angle, max_angle))

                # 求解逆运动学
                angles_start = self.mc.solve_inv_kinematics(start_coords, init_angles)
                angles_end = self.mc.solve_inv_kinematics(end_coords, angles_start)

                # 确保所有关节角度安全
                for i in range(6):
                    angles_start[i] = self.adjust_joint_angle(i + 1, angles_start[i])
                    angles_end[i] = self.adjust_joint_angle(i + 1, angles_end[i])

                # 计算各关节的运动量
                joint_movements = [abs(angles_end[i] - angles_start[i]) for i in range(6)]

                # 找出主导关节（运动量最大的优先关节）
                dominant_joint = None
                dominant_movement = 0

                # 优先从preferred_joints中选择主导关节
                for joint_id in preferred_joints:
                    movement = joint_movements[joint_id - 1]
                    if movement > dominant_movement:
                        dominant_joint = joint_id
                        dominant_movement = movement

                # 如果没有找到合适的主导关节，选择运动量最大的关节
                if dominant_joint is None:
                    dominant_joint = np.argmax(joint_movements) + 1
                    dominant_movement = joint_movements[dominant_joint - 1]

                # 计算其他关节的总运动量
                other_joints_movement = sum(joint_movements) - dominant_movement

                # 评估解的优劣：考虑主导关节是否在推荐列表中，以及运动效率
                preferred_bonus = 2.0 if dominant_joint in preferred_joints else 1.0
                if other_joints_movement > 0:
                    efficiency_score = dominant_movement / other_joints_movement
                else:
                    efficiency_score = 10.0  # 理想情况

                score = efficiency_score * preferred_bonus

                # 如果评分更高，则更新最佳解
                if score > best_score:
                    best_score = score
                    best_solution = (angles_start, angles_end, dominant_joint,
                                     dominant_movement, other_joints_movement, score)
                    print(f"找到更好解: 主导关节{dominant_joint}运动{dominant_movement:.1f}°, "
                          f"其他关节总运动{other_joints_movement:.1f}°, 评分{score:.2f}")

            except Exception as e:
                continue

        return best_solution

    def smart_grinding_path(self, start_coords, end_coords, speed=None):
        """
        智能打磨路径规划，根据路径方向选择最优运动策略

        Args:
            start_coords: 起点坐标 [x, y, z, rx, ry, rz]
            end_coords: 终点坐标 [x, y, z, rx, ry, rz]
            speed: 运动速度，如果为None则使用默认速度
        """
        if speed is None:
            speed = self.default_speed

        print("=== 开始智能打磨路径规划 ===")

        # 1. 分析路径方向，确定推荐的主导关节
        path_analysis = self.analyze_path_direction(start_coords, end_coords)
        preferred_joints = path_analysis['recommended_joints']

        print(f"推荐主导关节: {preferred_joints}")

        # 2. 检查当前各关节角度是否安全
        current_angles = self.mc.get_angles()
        for i, angle in enumerate(current_angles):
            joint_id = i + 1
            if not self.is_joint_safe(joint_id, angle):
                print(f"警告: 关节{joint_id}角度 {angle}° 超出安全范围")
                safe_angle = self.adjust_joint_angle(joint_id, angle)
                print(f"调整关节{joint_id}到安全位置: {safe_angle}°")
                self.mc.send_angle(joint_id, safe_angle, speed)
                time.sleep(1)

        # 3. 寻找最优运动解决方案
        solution = self.find_optimal_movement_solution(start_coords, end_coords, preferred_joints)

        if solution is None:
            print("警告：未找到理想解，使用默认逆运动学")
            current_angles = self.mc.get_angles()
            angles_start = self.mc.solve_inv_kinematics(start_coords, current_angles)
            angles_end = self.mc.solve_inv_kinematics(end_coords, angles_start)

            # 确保所有关节角度安全
            for i in range(6):
                angles_start[i] = self.adjust_joint_angle(i + 1, angles_start[i])
                angles_end[i] = self.adjust_joint_angle(i + 1, angles_end[i])

            # 默认使用关节1作为主导
            dominant_joint = 1
            dominant_movement = abs(angles_end[0] - angles_start[0])
            other_joints_movement = sum([abs(angles_end[i] - angles_start[i]) for i in range(1, 6)])
        else:
            angles_start, angles_end, dominant_joint, dominant_movement, other_joints_movement, score = solution

        print(f"起点角度: {[f'{a:.1f}' for a in angles_start]}")
        print(f"终点角度: {[f'{a:.1f}' for a in angles_end]}")
        print(f"主导关节{dominant_joint}运动: {dominant_movement:.1f}°")
        print(f"其他关节总运动: {other_joints_movement:.1f}°")

        # 4. 根据运动量选择策略
        if other_joints_movement < 15:  # 其他关节运动很小
            print(f"方案1: 其他关节运动很小，直接执行关节空间运动 (主导关节{dominant_joint})")
            self.mc.sync_send_angles(angles_start, speed, timeout=15)
            self.mc.sync_send_angles(angles_end, speed, timeout=15)

        elif other_joints_movement < 40:  # 其他关节运动中等
            print(f"方案2: 其他关节运动中等，使用分步优化策略 (主导关节{dominant_joint})")

            # 分步移动，优先移动主导关节
            steps = 3
            for step in range(steps):
                fraction = (step + 1) / steps
                intermediate_angles = []
                for i in range(6):
                    start_angle = angles_start[i]
                    end_angle = angles_end[i]
                    # 主导关节线性移动，其他关节缓动
                    if i + 1 == dominant_joint:
                        intermediate_angle = start_angle + (end_angle - start_angle) * fraction
                    else:
                        # 其他关节使用缓动函数，减少突变
                        ease_fraction = 1 - (1 - fraction) ** 2  # 缓入缓出
                        intermediate_angle = start_angle + (end_angle - start_angle) * ease_fraction

                    intermediate_angles.append(intermediate_angle)

                print(f"步骤{step + 1}: 角度{[f'{a:.1f}' for a in intermediate_angles]}")
                self.mc.sync_send_angles(intermediate_angles, speed, timeout=10)
                time.sleep(0.5)

        else:  # 其他关节运动较大
            print(f"方案3: 其他关节运动较大，使用主导关节优先策略 (主导关节{dominant_joint})")

            # 步骤1: 调整其他关节到近似位置，保持主导关节接近起点
            print("步骤1: 调整其他关节到近似位置")
            intermediate_angles = angles_end.copy()
            intermediate_angles[dominant_joint - 1] = angles_start[dominant_joint - 1]
            self.mc.sync_send_angles(intermediate_angles, speed // 2, timeout=15)
            time.sleep(1)

            # 步骤2: 主要依靠主导关节运动到终点
            print(f"步骤2: 主要依靠关节{dominant_joint}运动到终点")
            self.mc.send_angle(dominant_joint, angles_end[dominant_joint - 1], speed)
            time.sleep(1)
            while self.mc.is_moving():
                time.sleep(0.1)

            # 步骤3: 微调其他关节到精确位置
            print("步骤3: 微调其他关节到精确位置")
            self.mc.sync_send_angles(angles_end, speed // 3, timeout=10)

        # 验证最终位置
        final_coords = self.mc.get_coords()
        final_angles = self.mc.get_angles()
        print(f"最终坐标: {[f'{c:.1f}' for c in final_coords]}")
        print(f"最终角度: {[f'{a:.1f}' for a in final_angles]}")

        # 检查各关节角度安全
        for i, angle in enumerate(final_angles):
            joint_id = i + 1
            if not self.is_joint_safe(joint_id, angle):
                print(f"警告: 关节{joint_id}最终角度 {angle}° 超出安全范围!")
                safe_angle = self.adjust_joint_angle(joint_id, angle)
                self.mc.send_angle(joint_id, safe_angle, speed // 2)
                time.sleep(0.5)

        # 检查位置精度
        target_angles_diff = [abs(final_angles[i] - angles_end[i]) for i in range(6)]
        max_angle_error = max(target_angles_diff)
        print(f"最大角度误差: {max_angle_error:.1f}°")

        if max_angle_error > 5:
            print("进行微调...")
            self.mc.sync_send_angles(angles_end, speed // 3, timeout=5)

        print("=== 智能打磨路径规划完成 ===")

    def plan_grinding_task(self, center_coords, path_length=20.0, path_direction='x', tool_orientation=None,
                           speed=None):
        """
        完整的打磨任务规划函数
        根据中心点坐标自动生成打磨路径，并执行智能路径规划

        参数:
        - center_coords: 打磨路径的中心点坐标 [x, y, z, rx, ry, rz]
        - path_length: 打磨路径的长度（毫米），默认20mm
        - path_direction: 打磨路径的方向，可选 'x', 'y', 'z' 或自定义向量 [dx, dy, dz]
        - tool_orientation: 工具姿态，如果为None则使用中心点的姿态
        - speed: 运动速度，如果为None则使用默认速度
        """
        if speed is None:
            speed = self.default_speed

        print("=== 开始打磨任务规划 ===")
        print(f"中心点坐标: {center_coords}")
        print(f"路径长度: {path_length}mm")
        print(f"路径方向: {path_direction}")

        # 1. 生成打磨路径的起点和终点
        start_coords, end_coords = self.generate_grinding_path(
            center_coords, path_length, path_direction, tool_orientation
        )

        # 2. 执行智能路径规划
        try:
            self.mc.power_on()
            # self.smart_grinding_path(start_coords, end_coords, speed)
            self.mc.send_coords(start_coords, speed, 1)
            time.sleep(2)
            self.mc.send_coords(end_coords, speed, 1)
            time.sleep(2)
        except Exception as e:
            print(f"智能规划失败: {e}")
            print("尝试备选方案...")
            # 备选方案：直接使用坐标空间移动
            # self.mc.send_coords(start_coords, speed, 0)
            # time.sleep(2)
            # self.mc.send_coords(end_coords, speed, 0)
            # time.sleep(2)

        print("=== 打磨任务完成 ===")

    # 保留其他原有方法...
    def set_default_speed(self, speed):
        """设置默认运动速度"""
        self.default_speed = speed
        print(f"默认速度设置为: {speed}")

    def get_current_angles(self):
        """获取当前关节角度"""
        return self.mc.get_angles()

    def get_current_coords(self):
        """获取当前坐标"""
        return self.mc.get_coords()

    def is_moving(self):
        """检查机械臂是否在运动"""
        return self.mc.is_moving()

    def release(self):
        """释放资源"""
        print("MyCobot控制器资源释放")


# 使用示例
if __name__ == "__main__":
    # 创建MyCobot实例
    mycobot = MyCobot280('/dev/ttyAMA0', 1000000)

    # 创建控制器实例
    controller = MyCobotGrindingController(mycobot)

    # # 示例1: X方向路径（将自动选择关节2或3作为主导）
    # center_point = [187.3, 63.7, 195.6, 0, 180, 0]
    # controller.plan_grinding_task(center_point, path_length=100.0, path_direction='x', speed=3)
    #
    # # 示例2: Y方向路径（将自动选择关节1作为主导）
    # center_point = [187.3, 63.7, 195.6, 0, 180, 0]
    # controller.plan_grinding_task(center_point, path_length=100.0, path_direction='y', speed=3)
    #
    # # 示例3: Z方向路径（将自动选择关节2或3作为主导）
    # center_point = [187.3, 63.7, 195.6, 0, 180, 0]
    # controller.plan_grinding_task(center_point, path_length=50.0, path_direction='z', speed=3)