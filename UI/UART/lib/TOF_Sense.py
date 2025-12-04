# /*****************************************************************************
# * | File        :	  tof_sense.py
# * | Author      :   Waveshare team
# * | Function    :   TOF drive function
# * | Info        :
# *----------------
# * | This version:   V1.0
# * | Date        :   2024-09-11
# * | Info        :   
# ******************************************************************************
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documnetation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to  whom the Software is
# furished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS OR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#

import serial 
import time

TOF_FRAME_HEADER = 0x57  #Define frame header 定义帧头
TOF_FUNCTION_MARK = 0x00 #Define function code 定义功能码

#Store decoded data 存放解码后的数据
TOF_system_time = 0     #The time after the TOF module is powered on, unit: ms TOF模块上电后经过的时间，单位：ms
TOF_distance = 0        #The distance output by the TOF module, unit: mm TOF模块输出的距离，单位:mm
TOF_status = 0          #The distance status indication output by the TOF module: 0 is invalid, 1 is valid TOF模块输出的距离状态指示:0为无效,1为有效
TOF_signal_strength = 0 #The signal strength output by the TOF module TOF模块输出的信号强度
TOF_range_precision = 0 #The repeatability accuracy reference value output by the TOF module is invalid for C, D and Mini types. Unit: cm TOF模块输出的重复测距精度参考值，对于C型,D型和Mini型是无效的，单位:cm

TOF_rx_data=[0] * 16 #Create a list with 16 members 创建一个拥有16个成员的列表
TOF_tx_data=[0x57,0x10,0xff,0xff,0x00,0xff,0xff,0x63] #Query the command with ID 0 查询ID为0的命令


class TOF_Sense():
    #Open the Raspberry Pi serial port device 打开树莓派串口设备
    def __init__(self, dev = '/dev/ttyS0',baud = 921600):
        self.TOF_peek = 0 #Temporary storage of data 临时存放数据

        self.count_i = 0 #Loop count variable 循环计数变量
        self.check_sum = 0 #Checksum 校验和
        self.ser = serial.Serial(dev,baud)
        self.ser.flushInput()#Clear the serial port input register 清空串口输入寄存器
        self.distance = 0  # 存储当前距离值

    #Test active output mode 测试主动输出模式
    def TOF_Active_Decoding(self):    
        if self.ser.inWaiting() > 0: #Waiting for serial port data 等待串口数据
            self.TOF_peek = ord(self.ser.read(1))  #Read a byte and convert it into an integer 读取一个字节并转换成整数
            if self.TOF_peek == TOF_FRAME_HEADER: #If it is a frame header, restart the loop count 如果是帧头,则重新开始循环计数
                self.count_i = 0
                TOF_rx_data[self.count_i] = self.TOF_peek #Store the read data into a tuple for later decoding 将读取到的数据存入元组中，用于后面解码使用
            else:
                TOF_rx_data[self.count_i] = self.TOF_peek #Store the read data into a tuple for later decoding 将读取到的数据存入元组中，用于后面解码使用

            self.count_i = self.count_i + 1 #Loop count +1 循环计数+1

            if self.count_i > 15:#If the number of received data is greater than 15, the count variable can be cleared and a decoding can be performed. 接收数量大于15,则可以将计数变量清零并进行一次解码
                self.count_i = 0
                for i in range (0,15):
                    self.check_sum = (self.check_sum + TOF_rx_data[i]) & 0xFF #Calculate the checksum and take the lowest byte 计算检验和并取最低一个字节

                #Determine whether the decoding is correct 判断解码是否正确
                if (TOF_rx_data[0] == TOF_FRAME_HEADER) and (TOF_rx_data[1] == TOF_FUNCTION_MARK) and (self.check_sum == TOF_rx_data[15]):
                    print("TOF id is: "+ str(TOF_rx_data[3]))  #ID of the TOF module TOF 模块的 ID

                    TOF_system_time = TOF_rx_data[4] | TOF_rx_data[5]<<8 | TOF_rx_data[6]<<16 | TOF_rx_data[7]<<24
                    print("TOF system time is: "+str(TOF_system_time)+'ms') #The time after the TOF module is powered on TOF模块上电后经过的时间        

                    TOF_distance = (TOF_rx_data[8]) | (TOF_rx_data[9]<<8) | (TOF_rx_data[10]<<16)
                    print("TOF distance is: "+str(TOF_distance)+'mm') #The distance output by the TOF module TOF模块输出的距离   
            
                    TOF_status = TOF_rx_data[11]
                    print("TOF status is: "+str(TOF_status)) #Distance status indication output by TOF module TOF模块输出的距离状态指示
            
                    TOF_signal_strength = TOF_rx_data[12] | TOF_rx_data[13]<<8
                    print("TOF signal strength is: "+str(TOF_signal_strength)) #The signal strength output by the TOF module TOF模块输出的信号强度

                    TOF_range_precision = TOF_rx_data[14]
                    print("TOF range precision is: "+str(TOF_range_precision)) #The repeatability accuracy reference value output by the TOF module is invalid for Type C, Type D and Mini. TOF模块输出的重复测距精度参考值，对于C型,D型和Mini型是无效的
                    
                    print("")
                    self.ser.flushInput() #Clear the serial port input register 清空串口输入寄存器
                    self.distance = TOF_distance
                else:
                    print("Verification failed.")
            self.check_sum = 0 #Clear Checksum 清空校验和
        else:
            print("The serial port does not receive data.") 

    #Test query output mode 测试查询输出模式
    def TOF_Inquire_Decoding(self,id):
        TOF_tx_data[4] = id #Add the ID you want to query to the command 将需要查询的ID添加到命令中
        TOF_tx_data[7] = id + 0x63 #Update Checksum 更新校验和

        self.ser.flushInput() #Clear the serial port buffer 清空串口缓存
        self.ser.write(bytearray(TOF_tx_data)) #Start query 开始查询
        time.sleep(0.01) #Waiting for the sensor to return data 等待传感器返回数据
        TOF_rx_data = list(self.ser.read(16)) #Reading sensor data 读取传感器数据

        for i in range (0,15):
            self.check_sum = (self.check_sum + TOF_rx_data[i]) & 0xFF #Calculate the checksum and take the lowest byte 计算检验和并取最低一个字节

        #Determine whether the decoding is correct 判断解码是否正确
        if (TOF_rx_data[0] == TOF_FRAME_HEADER) and (TOF_rx_data[1] == TOF_FUNCTION_MARK) and (self.check_sum == TOF_rx_data[15]):
            print("TOF id is: "+ str(TOF_rx_data[3]))  #ID of the TOF module TOF 模块的 ID

            TOF_system_time = TOF_rx_data[4] | TOF_rx_data[5]<<8 | TOF_rx_data[6]<<16 | TOF_rx_data[7]<<24
            print("TOF system time is: "+str(TOF_system_time)+'ms') #The time after the TOF module is powered on TOF模块上电后经过的时间        

            TOF_distance = (TOF_rx_data[8]) | (TOF_rx_data[9]<<8) | (TOF_rx_data[10]<<16)
            print("TOF distance is: "+str(TOF_distance)+'mm') #The distance output by the TOF module TOF模块输出的距离   
    
            TOF_status = TOF_rx_data[11]
            print("TOF status is: "+str(TOF_status)) #Distance status indication output by TOF module TOF模块输出的距离状态指示
    
            TOF_signal_strength = TOF_rx_data[12] | TOF_rx_data[13]<<8
            print("TOF signal strength is: "+str(TOF_signal_strength)) #The signal strength output by the TOF module TOF模块输出的信号强度

            TOF_range_precision = TOF_rx_data[14]
            print("TOF range precision is: "+str(TOF_range_precision)) #The repeatability accuracy reference value output by the TOF module is invalid for Type C, Type D and Mini. TOF模块输出的重复测距精度参考值，对于C型,D型和Mini型是无效的
            
            print("")
            self.ser.flushInput() #Clear the serial port input register 清空串口输入寄存器
            self.distance = TOF_distance
        else:
            print("Verification failed.")
        self.check_sum = 0 #Clear Checksum 清空校验和

    def get_distance(self, sensor_id=0):
        """
        获取距离测量值（毫米）
        
        参数:
            sensor_id: 传感器ID（默认为0）
        
        返回:
            距离值（毫米）或None（失败时）
        """
        try:
            # 使用查询模式获取数据
            self.TOF_Inquire_Decoding(sensor_id)
            
            # 返回当前距离值
            return self.distance
        except Exception as e:
            print(f"获取距离时出错: {e}")
            return None