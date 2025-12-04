from pymycobot import MyCobot280
import time
import math
import numpy as np


class MyCobotGrindingController:
    """
    MyCobot280 打磨路径控制类
    专门用于控制机械臂进行打磨作业，特别优化一轴运动
    """

    def __init__(self, port='/dev/ttyAMA0', baudrate=1000000):
        """
        初始化机械臂控制器

        Args:
            port: 串口地址
            baudrate: 波特率
        """
        self.mc = MyCobot280(port, baudrate)

        # 一轴角度限制
        self.JOINT1_MIN = -55  # 一轴最小角度
        self.JOINT1_MAX = 85  # 一轴最大角度
        self.JOINT1_AVOID = 85  # 需要避开的特定角度

        # 默认运动速度
        self.default_speed = 5

        print(f"MyCobot280 控制器初始化完成，端口: {port}, 波特率: {baudrate}")

    def is_joint1_safe(self, angle):
        """检查一轴角度是否在安全范围内"""
        # 使用84.9而不是85来确保不等于85
        return self.JOINT1_MIN <= angle <= 84.9

    def adjust_joint1_angle(self, angle):
        """调整一轴角度到安全范围内"""
        if angle < self.JOINT1_MIN:
            return self.JOINT1_MIN
        elif angle > 84.9:  # 避免等于85
            return 84.9
        else:
            return angle

    def find_minimal_movement_solution(self, start_coords, end_coords, max_attempts=30):
        """
        寻找最小化其他关节运动的解决方案
        目标是让一轴完成大部分运动，其他关节尽量保持不动

        Args:
            start_coords: 起点坐标 [x, y, z, rx, ry, rz]
            end_coords: 终点坐标 [x, y, z, rx, ry, rz]
            max_attempts: 最大尝试次数

        Returns:
            tuple: (angles_start, angles_end, joint1_movement, other_joints_movement, score) 或 None
        """
        current_angles = self.mc.get_angles()
        best_solution = None
        min_other_joints_movement = float('inf')

        print("寻找最小化其他关节运动的解决方案...")

        for attempt in range(max_attempts):
            try:
                # 使用不同的初始角度尝试
                if attempt == 0:
                    init_angles = current_angles
                else:
                    # 生成随机的初始角度，但确保一轴在安全范围内
                    init_angles = [
                        np.random.uniform(self.JOINT1_MIN, 84.9),  # 一轴在安全范围内
                        np.random.uniform(-60, 60),  # 二轴
                        np.random.uniform(-120, 0),  # 三轴
                        np.random.uniform(-90, 90),  # 四轴
                        np.random.uniform(-90, 90),  # 五轴
                        np.random.uniform(-90, 90)  # 六轴
                    ]

                # 求解逆运动学
                angles_start = self.mc.solve_inv_kinematics(start_coords, init_angles)
                angles_end = self.mc.solve_inv_kinematics(end_coords, angles_start)

                # 确保一轴角度安全
                angles_start[0] = self.adjust_joint1_angle(angles_start[0])
                angles_end[0] = self.adjust_joint1_angle(angles_end[0])

                # 计算其他关节的总运动量（不包括一轴）
                other_joints_movement = sum([abs(angles_end[i] - angles_start[i]) for i in range(1, 6)])

                # 计算一轴的运动量
                joint1_movement = abs(angles_end[0] - angles_start[0])

                # 评估解的优劣：我们希望一轴运动大，其他关节运动小
                if other_joints_movement > 0:
                    score = joint1_movement / other_joints_movement
                else:
                    score = float('inf')  # 如果其他关节不动，这是理想情况

                # 如果其他关节运动量更小，或者评分更高，则更新最佳解
                if other_joints_movement < min_other_joints_movement or score > 10:
                    min_other_joints_movement = other_joints_movement
                    best_solution = (angles_start, angles_end, joint1_movement, other_joints_movement, score)
                    print(
                        f"找到更好解: 一轴运动{joint1_movement:.1f}°, 其他关节总运动{other_joints_movement:.1f}°, 评分{score:.2f}")

            except Exception as e:
                continue

        return best_solution

    def optimize_for_straight_line_joint1(self, start_coords, end_coords, num_points=10):
        """
        优化路径，使一轴运动尽可能直线
        通过在一轴运动路径上插入中间点，优化其他关节的位置

        Args:
            start_coords: 起点坐标
            end_coords: 终点坐标
            num_points: 路径点数

        Returns:
            list: 路径点角度列表
        """
        print("优化一轴直线运动路径...")

        # 获取起点和终点的关节角度
        current_angles = self.mc.get_angles()
        angles_start = self.mc.solve_inv_kinematics(start_coords, current_angles)
        angles_end = self.mc.solve_inv_kinematics(end_coords, angles_start)

        # 确保一轴角度安全
        angles_start[0] = self.adjust_joint1_angle(angles_start[0])
        angles_end[0] = self.adjust_joint1_angle(angles_end[0])

        # 计算一轴需要运动的总角度
        joint1_total_movement = angles_end[0] - angles_start[0]

        # 创建路径点列表
        path_points = []

        # 起点
        path_points.append(angles_start)

        # 中间点 - 主要依靠一轴运动
        for i in range(1, num_points):
            # 计算当前一轴位置（线性插值）
            fraction = i / num_points
            current_joint1 = angles_start[0] + joint1_total_movement * fraction

            # 尝试找到使其他关节运动最小的解
            best_intermediate = None
            min_other_movement = float('inf')

            # 尝试几种不同的其他关节配置
            for attempt in range(5):
                try:
                    # 使用不同的初始角度
                    if attempt == 0:
                        # 使用起点和终点的平均值作为其他关节的初始猜测
                        init_other = [(angles_start[j] + angles_end[j]) / 2 for j in range(1, 6)]
                    else:
                        # 随机初始角度
                        init_other = [
                            np.random.uniform(-60, 60),  # 二轴
                            np.random.uniform(-120, 0),  # 三轴
                            np.random.uniform(-90, 90),  # 四轴
                            np.random.uniform(-90, 90),  # 五轴
                            np.random.uniform(-90, 90)  # 六轴
                        ]

                    # 固定一轴角度，优化其他关节
                    fixed_joint1 = current_joint1

                    # 使用当前点作为初始猜测
                    init_angles = [fixed_joint1] + init_other

                    # 求解逆运动学，但固定一轴
                    target_angles = self.mc.solve_inv_kinematics(end_coords, init_angles)

                    # 固定一轴，只调整其他关节
                    target_angles[0] = fixed_joint1

                    # 计算其他关节的运动量（相对于起点）
                    other_movement = sum([abs(target_angles[j] - angles_start[j]) for j in range(1, 6)])

                    if other_movement < min_other_movement:
                        min_other_movement = other_movement
                        best_intermediate = target_angles.copy()

                except Exception as e:
                    continue

            if best_intermediate is not None:
                path_points.append(best_intermediate)
                print(
                    f"路径点 {i}/{num_points}: 一轴={best_intermediate[0]:.1f}°, 其他关节运动={min_other_movement:.1f}°")

        # 终点
        path_points.append(angles_end)

        return path_points

    def safe_joint1_movement(self, target_angle, speed=None):
        """
        安全地移动一轴，确保不超出限制

        Args:
            target_angle: 目标角度
            speed: 运动速度，如果为None则使用默认速度
        """
        if speed is None:
            speed = self.default_speed

        safe_angle = self.adjust_joint1_angle(target_angle)

        # 获取当前一轴角度
        current_angles = self.mc.get_angles()
        current_joint1 = current_angles[0]

        # 检查是否需要调整
        if abs(safe_angle - target_angle) > 0.1:
            print(f"警告: 一轴目标角度 {target_angle}° 超出安全范围，调整为 {safe_angle}°")

        # 直接移动到目标角度（不分段）
        print(f"一轴直接移动: {current_joint1:.1f}° -> {safe_angle:.1f}°")
        self.mc.send_angle(1, safe_angle, speed)

        # 等待运动完成
        time.sleep(1)
        while self.mc.is_moving():
            time.sleep(0.1)

        # 最终验证
        final_angles = self.mc.get_angles()
        if not self.is_joint1_safe(final_angles[0]):
            print(f"错误: 一轴最终位置超出安全范围: {final_angles[0]}°")
            # 强制调整到安全位置
            safe_final = self.adjust_joint1_angle(final_angles[0])
            self.mc.send_angle(1, safe_final, speed // 2)
            time.sleep(1)

    def joint1_dominant_grinding_path(self, start_coords, end_coords, speed=None):
        """
        一轴主导的打磨路径规划

        Args:
            start_coords: 起点坐标 [x, y, z, rx, ry, rz]
            end_coords: 终点坐标 [x, y, z, rx, ry, rz]
            speed: 运动速度，如果为None则使用默认速度
        """
        if speed is None:
            speed = self.default_speed

        print("=== 开始一轴主导的打磨路径规划 ===")
        print(f"一轴工作范围: [{self.JOINT1_MIN}°, {84.9}°] (避开{self.JOINT1_AVOID}°)")

        # 1. 检查当前一轴角度是否安全
        current_angles = self.mc.get_angles()
        if not self.is_joint1_safe(current_angles[0]):
            print(f"警告: 当前一轴角度 {current_angles[0]}° 超出安全范围")
            print("先调整一轴到安全位置...")
            safe_angle = self.adjust_joint1_angle(current_angles[0])
            self.safe_joint1_movement(safe_angle, speed)

        # 2. 寻找最小化其他关节运动的解决方案
        solution = self.find_minimal_movement_solution(start_coords, end_coords)

        if solution is None:
            print("警告：未找到理想解，使用默认逆运动学")
            current_angles = self.mc.get_angles()
            angles_start = self.mc.solve_inv_kinematics(start_coords, current_angles)
            angles_end = self.mc.solve_inv_kinematics(end_coords, angles_start)

            # 确保一轴角度安全
            angles_start[0] = self.adjust_joint1_angle(angles_start[0])
            angles_end[0] = self.adjust_joint1_angle(angles_end[0])

            joint1_movement = abs(angles_end[0] - angles_start[0])
            other_joints_movement = sum([abs(angles_end[i] - angles_start[i]) for i in range(1, 6)])
        else:
            angles_start, angles_end, joint1_movement, other_joints_movement, score = solution

        print(f"起点角度: {[f'{a:.1f}' for a in angles_start]}")
        print(f"终点角度: {[f'{a:.1f}' for a in angles_end]}")
        print(f"一轴运动: {joint1_movement:.1f}°")
        print(f"其他关节总运动: {other_joints_movement:.1f}°")

        # 3. 根据其他关节运动量选择策略
        if other_joints_movement < 10:  # 其他关节运动很小
            print("方案1: 其他关节运动很小，直接执行关节空间运动")

            # 直接移动到起点
            print("移动到起点...")
            self.mc.sync_send_angles(angles_start, speed, timeout=15)

            # 直接移动到终点（主要是一轴运动）
            print("移动到终点（主要是一轴运动）...")
            self.mc.sync_send_angles(angles_end, speed, timeout=15)

        elif other_joints_movement < 30:  # 其他关节运动中等
            print("方案2: 其他关节运动中等，使用优化路径")

            # 使用优化路径，尽量减少其他关节运动
            path_points = self.optimize_for_straight_line_joint1(start_coords, end_coords, num_points=5)

            # 执行路径
            print("执行优化路径...")
            for i, point in enumerate(path_points):
                print(f"路径点 {i + 1}/{len(path_points)}: 角度{[f'{a:.1f}' for a in point]}")

                # 使用同步移动确保到达每个点
                if i == 0:
                    # 第一个点使用正常速度
                    self.mc.sync_send_angles(point, speed, timeout=10)
                else:
                    # 中间点使用较慢速度，减少抖动
                    self.mc.sync_send_angles(point, speed // 2, timeout=8)

                # 短暂暂停，确保稳定
                time.sleep(0.5)

        else:  # 其他关节运动较大
            print("方案3: 其他关节运动较大，使用分步策略")

            # 步骤1: 先调整其他关节到近似位置，保持一轴接近起点
            print("步骤1: 调整其他关节到近似位置")

            # 创建中间角度，其他关节接近终点，一轴保持起点
            intermediate_angles = angles_end.copy()
            intermediate_angles[0] = angles_start[0]  # 一轴保持起点角度

            self.mc.sync_send_angles(intermediate_angles, speed // 2, timeout=15)
            time.sleep(1)

            # 步骤2: 主要依靠一轴运动到终点
            print("步骤2: 主要依靠一轴运动到终点")
            self.safe_joint1_movement(angles_end[0], speed)

            # 步骤3: 微调其他关节到精确位置
            print("步骤3: 微调其他关节到精确位置")
            self.mc.sync_send_angles(angles_end, speed // 3, timeout=10)

        # 验证最终位置
        final_coords = self.mc.get_coords()
        final_angles = self.mc.get_angles()
        print(f"最终坐标: {[f'{c:.1f}' for c in final_coords]}")
        print(f"最终角度: {[f'{a:.1f}' for a in final_angles]}")

        # 特别检查一轴角度
        if not self.is_joint1_safe(final_angles[0]):
            print(f"错误: 最终一轴角度 {final_angles[0]}° 超出安全范围!")
            safe_final = self.adjust_joint1_angle(final_angles[0])
            print(f"调整到安全角度: {safe_final}°")
            self.safe_joint1_movement(safe_final, speed // 2)
        else:
            print("一轴角度在安全范围内")

        # 检查位置精度
        target_angles_diff = [abs(final_angles[i] - angles_end[i]) for i in range(6)]
        max_angle_error = max(target_angles_diff)
        print(f"最大角度误差: {max_angle_error:.1f}°")

        if max_angle_error > 5:
            print("警告: 位置误差较大，进行微调")
            # 微调时也要确保一轴安全
            safe_angles_end = angles_end.copy()
            safe_angles_end[0] = self.adjust_joint1_angle(safe_angles_end[0])
            self.mc.sync_send_angles(safe_angles_end, speed // 3, timeout=5)

        print("=== 打磨路径规划完成 ===")

    def alternative_cartesian_approach(self, start_coords, end_coords, speed=None):
        """
        备选方案：在笛卡尔空间实现近似直线运动，但尽量减少其他关节运动

        Args:
            start_coords: 起点坐标
            end_coords: 终点坐标
            speed: 运动速度，如果为None则使用默认速度
        """
        if speed is None:
            speed = self.default_speed

        print("=== 使用笛卡尔空间备选方案 ===")
        print(f"一轴工作范围: [{self.JOINT1_MIN}°, {84.9}°]")

        # 1. 移动到起点
        print("移动到起点...")

        # 先通过逆运动学找到安全的角度
        current_angles = self.mc.get_angles()
        angles_start = self.mc.solve_inv_kinematics(start_coords, current_angles)
        angles_start[0] = self.adjust_joint1_angle(angles_start[0])

        # 使用关节空间移动到起点，确保一轴安全
        self.safe_joint1_movement(angles_start[0], speed)
        self.mc.sync_send_angles(angles_start, speed, timeout=15)

        # 2. 在笛卡尔空间分割路径，但尽量减少其他关节运动
        segments = 10  # 更多分段以获得更平滑的运动
        start_pos = np.array(start_coords)
        end_pos = np.array(end_coords)

        # 计算每个分段的其他关节运动量，确保最小化
        for i in range(1, segments + 1):
            fraction = i / segments

            # 线性插值计算中间点
            intermediate_coords = start_pos + (end_pos - start_pos) * fraction

            # 使用逆运动学求解，但尝试最小化其他关节运动
            best_angles = None
            min_other_movement = float('inf')

            for attempt in range(5):
                try:
                    # 使用不同的初始角度
                    if attempt == 0:
                        init_angles = angles_start
                    else:
                        init_angles = [
                            np.random.uniform(self.JOINT1_MIN, 84.9),
                            np.random.uniform(-60, 60),
                            np.random.uniform(-120, 0),
                            np.random.uniform(-90, 90),
                            np.random.uniform(-90, 90),
                            np.random.uniform(-90, 90)
                        ]

                    # 求解逆运动学
                    angles = self.mc.solve_inv_kinematics(list(intermediate_coords), init_angles)
                    angles[0] = self.adjust_joint1_angle(angles[0])

                    # 计算其他关节相对于起点的运动量
                    other_movement = sum([abs(angles[j] - angles_start[j]) for j in range(1, 6)])

                    if other_movement < min_other_movement:
                        min_other_movement = other_movement
                        best_angles = angles

                except Exception as e:
                    continue

            if best_angles is not None:
                print(f"段 {i}/{segments}: 一轴={best_angles[0]:.1f}°, 其他运动={min_other_movement:.1f}°")

                # 使用关节空间移动，确保主要是一轴运动
                self.mc.sync_send_angles(best_angles, speed // 2, timeout=5)

                # 检查一轴角度安全
                current_angles = self.mc.get_angles()
                if not self.is_joint1_safe(current_angles[0]):
                    print(f"警告: 一轴角度 {current_angles[0]}° 接近限制")
                    safe_angle = self.adjust_joint1_angle(current_angles[0])
                    self.safe_joint1_movement(safe_angle, speed // 2)
            else:
                print(f"段 {i}/{segments}: 逆运动学求解失败，使用坐标控制")
                # 备用方案：直接使用坐标控制
                self.mc.send_coords(list(intermediate_coords), speed // 3, 0)

        print("笛卡尔空间方案完成")

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
        """释放资源（如果需要的话）"""
        print("MyCobot控制器资源释放")


# 使用示例
if __name__ == "__main__":
    # 创建控制器实例
    controller = MyCobotGrindingController('/dev/ttyAMA0', 1000000)

    # 定义起点和终点坐标
    start_coords = [268.5787542987266, 46.51447156670595, 193.39731483179764, 0, 180, 0]
    end_coords = [241.2885594240246, -76.24241360652088, 193.0055581775897, 0, 180, 0]

    # 设置速度
    speed = 5

    try:
        # 执行一轴主导的打磨路径
        controller.joint1_dominant_grinding_path(start_coords, end_coords, speed)

        # 或者使用备选方案
        # controller.alternative_cartesian_approach(start_coords, end_coords, speed)

    except Exception as e:
        print(f"执行过程中发生错误: {e}")

    finally:
        # 释放资源
        controller.release()