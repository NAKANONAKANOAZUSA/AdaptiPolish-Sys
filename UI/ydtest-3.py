from threading import Thread
from pymycobot import MyCobot280
import time
import numpy as np
import serial
import crcmod


# 电机控制器类
class MotorController:
    def __init__(self, port='/dev/ttyCH343USB0', baudrate=115200, address=0x00):
        """电机控制器初始化"""
        try:
            self.ser = serial.Serial(
                port=port,
                baudrate=baudrate,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=0.1
            )
            self.connected = True
            print("电机控制器连接成功")
        except Exception as e:
            print(f"电机连接失败: {e}")
            self.connected = False

        self.address = address
        self.crc16 = crcmod.mkCrcFun(
            0x18005,
            rev=True,
            initCrc=0xFFFF,
            xorOut=0x0000
        )
        self.is_running = False

    def _create_command(self, function_code, register, value):
        """创建Modbus RTU命令帧"""
        frame = bytearray([
            self.address,
            function_code,
            (register >> 8) & 0xFF,
            register & 0xFF,
            (value >> 8) & 0xFF,
            value & 0xFF
        ])
        crc = self.crc16(frame)
        frame.append(crc & 0xFF)
        frame.append((crc >> 8) & 0xFF)
        return frame

    def send_command(self, command):
        """发送命令并获取响应"""
        if not self.connected:
            return None
        try:
            self.ser.write(command)
            time.sleep(0.05)
            return self.ser.read(8)
        except Exception as e:
            print(f"电机命令发送失败: {e}")
            return None

    def start(self):
        """启动电机"""
        if not self.connected:
            return None
        cmd = self._create_command(0x06, 0x0052, 0x0001)
        result = self.send_command(cmd)
        if result:
            self.is_running = True
            print("电机启动成功")
        return result

    def stop(self):
        """停止电机"""
        if not self.connected:
            return None
        cmd = self._create_command(0x06, 0x0050, 0x0001)
        result = self.send_command(cmd)
        if result:
            self.is_running = False
            print("电机关闭成功")
        return result

    def set_speed(self, speed):
        """设置电机转速(0-500)"""
        if not self.connected:
            return None
        if not 0 <= speed <= 500:
            raise ValueError("速度值必须在0-500范围内")
        cmd = self._create_command(0x06, 0x005A, speed)
        return self.send_command(cmd)

    def close(self):
        """关闭串口连接"""
        if self.connected:
            self.stop()
            self.ser.close()
            self.connected = False


# 给定的4x4变换矩阵
T = np.array([
    [-2.74280449e-02, -9.99601611e-01, -6.65752354e-03, 6.48263423e+01],
    [-9.99259510e-01, 2.72376602e-02, 2.71761376e-02, 2.04309745e+02],
    [-2.69839756e-02, 7.39798203e-03, -9.99608491e-01, 8.88548188e+02],
    [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
])


# 直接使用矩阵求逆
def transform_point_inverse(T, point):
    """使用逆矩阵变换点"""
    point_homogeneous = np.append(point, 1.0)
    T_inv = np.linalg.inv(T)
    robot_point_homogeneous = T_inv @ point_homogeneous
    robot_point = robot_point_homogeneous[:3] / robot_point_homogeneous[3]
    return robot_point


def move_to_point_clouds(point_cloud_list, motor_controller, speed=20, fixed_height=None, motor_speed=400):
    """
    将点云坐标列表转换为机器人坐标并使用直线运动模式依次运动到每个位置，在运动时开启电机

    Args:
        point_cloud_list: 点云坐标列表，每个元素为[x, y, z]格式的数组
        motor_controller: 电机控制器实例
        speed: 直线运动速度（1-100）
        fixed_height: 固定高度，如果为None则使用点云中的Z坐标
        motor_speed: 电机转速（0-500）
    """

    def get_compensation_value(x, y):
        max_abs_value = max(abs(x), abs(y))
        if max_abs_value <= 100:
            return 0
        elif max_abs_value <= 200:
            return 0
        else:
            return 0

    def limit_coordinate(value, min_val, max_val):
        return max(min(value, max_val), min_val)

    # 存储所有笛卡尔坐标点
    cartesian_points = []
    print(f"开始处理 {len(point_cloud_list)} 个点...")

    # 计算所有点的笛卡尔坐标
    for i, point_cloud in enumerate(point_cloud_list):
        print(f"计算第 {i + 1}/{len(point_cloud_list)} 个点的坐标...")

        # 将点云坐标转换为机器人坐标系
        robot_coords_inverse = transform_point_inverse(T, point_cloud)
        print("原机器人坐标:", robot_coords_inverse)

        compensation_value = get_compensation_value(robot_coords_inverse[0], robot_coords_inverse[1])
        z_compensation = 0

        # 应用补偿
        compensated_x = robot_coords_inverse[0] + compensation_value
        compensated_y = robot_coords_inverse[1] + compensation_value
        compensated_z = robot_coords_inverse[2] + z_compensation

        # 限制坐标范围
        x_limited = limit_coordinate(compensated_x, -281.45, 281.45)
        y_limited = limit_coordinate(compensated_y, -281.45, 281.45)
        z_limited = limit_coordinate(compensated_z, -70, 412.67)

        # 计算最终坐标
        coords = [x_limited + 10, y_limited, z_limited + 130 + 17.5, 0, 180, 0]

        if fixed_height is not None:
            coords[2] = fixed_height

        cartesian_points.append(coords)
        print(f"点 {i + 1}: 世界坐标 {point_cloud} -> 机器人坐标 {coords}")

    filtered_points = cartesian_points.copy()
    print(f"过滤后剩余 {len(filtered_points)} 个点")

    if len(filtered_points) == 0:
        print("没有有效的点需要运动")
        return

    # 使用直线运动模式执行所有点
    print(f"\n开始直线轨迹运动，共 {len(filtered_points)} 个点...")
    try:
        motor_controller.set_speed(0)
        print(f"电机速度设置为: {0}")
    except Exception as e:
        print(f"设置电机速度失败: {e}")
    # 先移动到初始位置
    print("准备运动到起始点...")
    mc.sync_send_angles([0.17, 31.2, -116.19, -13.27, 1.66, 116.63], 30, timeout=1)
    mc.sync_send_angles([39.55, -0.17, -59.76, -34.27, 2.54, 128.05], 30, timeout=1)
    result = mc.sync_send_coords(filtered_points[0], speed, mode=0, timeout=0.5)
    if result != 1:
        print("移动到起始点失败！")
        return

    time.sleep(0.5)

    # 然后使用直线运动模式依次运动到其他点，在运动时开启电机
    success_count = 0

    try:
        motor_controller.set_speed(motor_speed)
        motor_controller.stop()
        print(f"电机速度设置为: {motor_speed}")
    except Exception as e:
        print(f"设置电机速度失败: {e}")

    for i, coords in enumerate(filtered_points):
        if i == 0:  # 第一个点已经处理过
            success_count += 1
            continue

        print(f"运动到第 {i + 1}/{len(filtered_points)} 个点...")
        # 设置电机速度

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
    print(f"\n运动完成！成功到达 {success_count}/{len(filtered_points)} 个点")

    try:
        motor_controller.set_speed(0)
        print(f"电机速度设置为: {0}")
    except Exception as e:
        print(f"设置电机速度失败: {e}")

    # 返回初始位置
    mc.sync_send_angles([39.55, -0.17, -59.76, -34.27, 2.54, 128.05], 30, timeout=1)
    mc.sync_send_angles([0.17, 31.2, -116.19, -13.27, 1.66, 116.63], 30, timeout=1)


if __name__ == "__main__":
    try:
        # 初始化机械臂控制器
        mc = MyCobot280('/dev/ttyAMA0', 1000000)
        mc.power_on()

        # 初始化电机控制器
        motor = MotorController()

        # 运动参数
        offset_x = 20
        offset_y = 5
        height = 790
        motor_speed = 400  # 电机转速

        # 点云坐标列表
        point_clouds = [
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

        # 执行运动（传入电机控制器实例）
        move_to_point_clouds(point_clouds, motor, motor_speed=motor_speed)

    except Exception as e:
        print(f"发生错误: {e}")