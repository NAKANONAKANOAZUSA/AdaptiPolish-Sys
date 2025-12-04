#coding: UTF-8
import sys 
import time
import os

# 获取当前脚本所在目录的上级目录（UART目录）
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

# 添加lib目录到Python路径
sys.path.append(os.path.join(parent_dir, 'lib'))

# 导入TOF_Sense模块
from TOF_Sense import TOF_Sense

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





# # 获取当前脚本所在目录（项目目录）
# current_dir = os.path.dirname(os.path.abspath(__file__))

# # 添加lib目录到Python路径
# sys.path.append(os.path.join(current_dir, 'lib'))

# # 导入TOF_Sense模块
# from TOF_Sense import TOF_Sense

# # 初始化TOF传感器
# tof = TOF_Sense('COM3', 921600) 

# # 主循环
# try:
#     while True:
#         # 使用主动解码模式
#         tof.TOF_Active_Decoding()
        
#         # 或者使用查询模式（取消下面一行的注释）
#         # tof.TOF_Inquire_Decoding(0)  # 传入传感器ID（例如0）
        
#         time.sleep(0.02)  # 1000毫秒延迟（50Hz刷新率）

# except KeyboardInterrupt:
#     print("程序已退出")