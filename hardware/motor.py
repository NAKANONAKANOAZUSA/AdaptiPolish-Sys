import serial
import crcmod
import time
class MotorController:
    def __init__(self, port='/dev/ttyCH343USB0', baudrate=115200, address=0x00):
        """电机控制器初始化"""
        # 创建串口连接对象
        try:
            self.ser = serial.Serial(
                port=port,                  # 串口设备路径
                baudrate=baudrate,          # 通信波特率
                bytesize=serial.EIGHTBITS,  # 数据位：8位
                parity=serial.PARITY_NONE,  # 校验位：无校验
                stopbits=serial.STOPBITS_ONE, # 停止位：1位
                timeout=0.1                 # 超时时间：0.1秒
            )
        except:

            print("电机连接失败！")

        self.address = address          # Modbus设备地址
        # 创建CRC16校验函数
        self.crc16 = crcmod.mkCrcFun(
            0x18005,     # CRC多项式
            rev=True,    # 反转输入字节
            initCrc=0xFFFF,  # 初始CRC值
            xorOut=0x0000    # 最终异或值
        )

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
        self.ser.write(command)
        time.sleep(0.05)
        return self.ser.read(8)

    def start(self):
        """启动电机"""
        cmd = self._create_command(0x06, 0x0052, 0x0001)
        return self.send_command(cmd)

    def stop(self):
        """复位电机"""
        cmd = self._create_command(0x06, 0x0050, 0x0001)
        return self.send_command(cmd)

    def forward(self):
        """正转"""
        cmd = self._create_command(0x06, 0x0054, 0x0001)
        return self.send_command(cmd)

    def reverse(self):
        """反转"""
        cmd = self._create_command(0x06, 0x0056, 0x0001)
        return self.send_command(cmd)

    def emergency_stop(self):   
        """急停"""
        cmd = self._create_command(0x06, 0x0058, 0x0001)
        return self.send_command(cmd)

    def set_speed(self, speed):
        """设置电机转速(0-500)"""
        if not 0 <= speed <= 500:
            raise ValueError("速度值必须在0-500范围内")
        cmd = self._create_command(0x06, 0x005A, speed)
        return self.send_command(cmd)

    def close(self):
        """关闭串口连接"""
        self.ser.close()