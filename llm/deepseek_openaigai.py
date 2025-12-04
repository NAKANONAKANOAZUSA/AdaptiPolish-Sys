from openai import OpenAI
import re

llm_model_path = 'deepseek-r1-1.5b'

class LlmModel:
    def __init__(self, model_path=llm_model_path, system_prompt=None):
        self._model = model_path
        self._client = OpenAI(
            base_url='http://127.0.0.1:11434/v1',
            api_key='ollama',  # required, but unused
        )
        
        self._system_prompt = """
        **角色设定：**
        你是机器人控制系统的智能助手，负责处理机器人操作指令和技术问题。
        
        **指令识别：**
        当用户发出操作指令时，请返回对应的指令代码：
        
        **核心机器人控制指令：**
        - 启动/停止摄像头: <command>TOGGLE_CAMERA</command>
        - 连接/断开机器人: <command>CONNECT_ROBOT</command> / <command>DISCONNECT_ROBOT</command>
        - 开始标定: <command>START_CALIBRATION</command>
        - 启动/停止打磨: <command>START_GRINDING</command> / <command>STOP_GRINDING</command>
        - 启动/停止语音识别: <command>START_SPEECH_RECOGNITION</command> / <command>STOP_SPEECH_RECOGNITION</command>
        
        **示教点管理指令：**
        - 保存当前位置: <command>SAVE_TEACH_POINT</command>
        - 移动到指定点: <command>MOVE_TO_POINT:[点名称]</command>
        - 删除示教点: <command>DELETE_POINT:[点名称]</command>
        - 执行所有点: <command>EXECUTE_ALL_POINTS</command>
        
        **机械臂控制指令：**
        - 移动到安全位置: <command>MOVE_TO_SAFE</command>
        - 移动到坐标: <command>MOVE_TO_COORD:[X],[Y],[Z]</command>
        - 暂停执行: <command>PAUSE_EXECUTION</command>
        - 恢复执行: <command>RESUME_EXECUTION</command>
        - 停止执行: <command>STOP_EXECUTION</command>
        
        **参数设置指令：**
        - 设置X偏移: <command>SET_OFFSET_X:[值]</command>
        - 设置Y偏移: <command>SET_OFFSET_Y:[值]</command>
        - 设置Z偏移: <command>SET_OFFSET_Z:[值]</command>
        - 设置循环次数: <command>SET_LOOPS:[次数]</command>
        - 设置速度: <command>SET_SPEED:[值]</command>
        
        **电机控制指令：**
        - 启动/停止电机: <command>MOTOR_START</command> / <command>MOTOR_STOP</command>
        - 正转/反转: <command>MOTOR_FORWARD</command> / <command>MOTOR_REVERSE</command>
        - 急停: <command>MOTOR_EMERGENCY</command>
        - 设置转速: <command>SET_MOTOR_SPEED:[转速]</command>
        
        **系统指令：**
        - 加载模型: <command>LOAD_MODEL</command>
        - 保存/加载标定: <command>SAVE_CALIBRATION</command> / <command>LOAD_CALIBRATION</command>
        - 退出系统: <command>SHUTDOWN</command>
        
        **回答风格：**
        1. 用中文回答，简洁明了
        2. 直接回答问题，避免无关内容
        3. 重点突出，优先回答核心问题
        4. 复杂问题分点回答
        5. 总回复长度不超过40字
        """

        # 预设回答模板
        self._response_templates = {
            # 核心指令确认
            "toggle_camera": "正在切换摄像头状态...",
            "connect_robot": "正在连接机器人...",
            "disconnect_robot": "正在断开机器人连接...",
            "start_calibration": "开始标定流程...",
            "start_grinding": "正在启动打磨操作...",
            "stop_grinding": "正在停止打磨操作...",
            "start_speech_recognition": "启动语音识别功能...",
            "stop_speech_recognition": "停止语音识别功能...",
            
            # 示教点管理
            "save_teach_point": "已保存当前位置为示教点",
            "move_to_point": "正在移动到指定点...",
            "delete_point": "已删除指定示教点",
            "execute_all_points": "开始执行所有示教点...",
            
            # 机械臂控制
            "move_to_safe": "正在移动到安全位置...",
            "move_to_coord": "正在移动到指定坐标...",
            "pause_execution": "已暂停执行",
            "resume_execution": "已恢复执行",
            "stop_execution": "已停止执行",
            
            # 参数设置
            "set_offset_x": "X偏移已设置",
            "set_offset_y": "Y偏移已设置",
            "set_offset_z": "Z偏移已设置",
            "set_loops": "循环次数已设置",
            "set_speed": "速度参数已设置",
            
            # 电机控制
            "motor_start": "电机已启动",
            "motor_stop": "电机已停止",
            "motor_forward": "电机正转中",
            "motor_reverse": "电机反转中",
            "motor_emergency": "电机已急停",
            "set_motor_speed": "电机转速已设置",
            
            # 系统指令
            "load_model": "正在加载模型...",
            "save_calibration": "标定参数已保存",
            "load_calibration": "标定参数已加载",
            "shutdown": "系统正在关闭..."
        }

    def generate(self, text):
        # 检查是否有匹配的预设回答
        preset_response = self._check_preset_responses(text)
        if preset_response:
            # 如果是预设回答，直接返回
            yield preset_response
            return

        # 否则使用LLM生成回答
        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": text}
        ]
        
        try:
            # 添加 think=False 参数禁用思考模
            stream = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                stream=True,
                #think=False  # 关键参数：禁用思考模式
            )
        except Exception as e:
            print(f"API调用错误: {e}")
            yield "系统错误，请稍后再试"
            return
        
        # 收集完整响应
        full_response = ""
        for chunk in stream:
            c = chunk.choices[0].delta.content or ""
            full_response += c
            yield c
        
        # 检查响应中是否包含指令
        command = self._extract_command(full_response)
        if command:
            # 直接返回指令，不额外生成内容
            yield f"<command>{command}</command>"

    def _check_preset_responses(self, text):
        """检查输入是否匹配预设回答模板"""
        text_lower = text.lower().strip()
    
        # 指令检测 - 使用更灵活的匹配方式
        if any(phrase in text_lower for phrase in ["摄像头", "摄像", "相机", "视频"]):
            return self._response_templates["toggle_camera"]
        
        # 机器人连接控制
        if any(phrase in text_lower for phrase in ["连接机器人", "机器人连接", "连接机械臂"]):
            return self._response_templates["connect_robot"]
        if any(phrase in text_lower for phrase in ["断开机器人", "断开连接", "机器人断开"]):
            return self._response_templates["disconnect_robot"]
        
        # 标定指令
        if any(phrase in text_lower for phrase in ["标定", "校准", "摄像头标定"]):
            return self._response_templates["start_calibration"]
        
        # 打磨控制
        if any(phrase in text_lower for phrase in ["开始打磨", "启动打磨", "运行打磨"]):
            return self._response_templates["start_grinding"]
        if any(phrase in text_lower for phrase in ["停止打磨", "结束打磨", "暂停打磨"]):
            return self._response_templates["stop_grinding"]
        
        # 示教点操作
        if any(phrase in text_lower for phrase in ["保存点", "记录位置", "保存位置"]):
            return self._response_templates["save_teach_point"]
        if any(phrase in text_lower for phrase in ["移动点", "到点", "示教点"]):
            point_name = self._extract_point_name(text)
            return f"移动到点 {point_name}" if point_name else self._response_templates["move_to_point"]
        
        # 坐标移动
        if any(phrase in text_lower for phrase in ["移动坐标", "到坐标", "移动到位置"]):
            coords = self._extract_coordinates(text)
            return f"移动到坐标 {coords}" if coords else self._response_templates["move_to_coord"]
        
        # 参数设置
        if any(phrase in text_lower for phrase in ["设置偏移", "修改偏移"]):
            axis, value = self._extract_axis_value(text)
            if axis and value is not None:
                return f"{axis.upper()}偏移设置为 {value}"
            return self._response_templates["set_offset_x"]
        
        # 电机控制
        if any(phrase in text_lower for phrase in ["启动电机", "开启电机"]):
            return self._response_templates["motor_start"]
        if any(phrase in text_lower for phrase in ["停止电机", "关闭电机"]):
            return self._response_templates["motor_stop"]
        if any(phrase in text_lower for phrase in ["电机正转", "正转"]):
            return self._response_templates["motor_forward"]
        if any(phrase in text_lower for phrase in ["电机反转", "反转"]):
            return self._response_templates["motor_reverse"]
        
        # 如果未匹配任何预设指令，返回None让LLM处理
        return None
    
    def _extract_point_name(self, text):
        """从文本中提取示教点名称"""
        # 示例实现 - 实际应用中可以使用更复杂的方法
        match = re.search(r'(点|位置)(\w+)', text)
        return match.group(2) if match else None
    
    def _extract_coordinates(self, text):
        """从文本中提取坐标值"""
        # 示例实现 - 实际应用中可以使用更复杂的方法
        numbers = re.findall(r'\d+', text)
        return ",".join(numbers[:3]) if len(numbers) >= 3 else None
    
    def _extract_axis_value(self, text):
        """从文本中提取轴和值"""
        # 示例实现
        axis_match = re.search(r'([xyz])轴?', text, re.IGNORECASE)
        value_match = re.search(r'(\d+(\.\d+)?)', text)
        
        axis = axis_match.group(1).lower() if axis_match else None
        value = float(value_match.group(1)) if value_match else None
        
        return axis, value
    
    def _extract_command(self, response):
        """从响应中提取指令"""
        # 使用正则表达式匹配指令格式
        match = re.search(r'<command>(.*?)</command>', response)
        if match:
            return match.group(1)
        return None
    
    def set_system_prompt(self, new_prompt):
        """动态修改系统提示词"""
        self._system_prompt = new_prompt
    
    def add_response_template(self, key, pattern, response):
        """添加新的响应模板"""
        self._response_templates[key] = response

if __name__ == '__main__':
    # 使用默认系统提示词
    llm_model = LlmModel()
    
    # 测试指令识别
    test_commands = [
        "请启动打磨操作",
        "我需要停止打磨",
        "现在进行标定",
        "连接机器人",
        "断开麦克风连接",
        "启动语音识别功能"
    ]
    
    for command in test_commands:
        print(f"用户: {command}")
        print("AI: ", end="", flush=True)
        
        # 收集响应
        response = ""
        detected_command = None
        
        for c in llm_model.generate(command):
            print(c, end='', flush=True)
            response += c
            
            # 检查是否包含指令标记
            if "<command>" in response:
                # 提取指令
                command_start = response.find("<command>") + len("<command>")
                command_end = response.find("</command>")
                if command_end != -1:
                    detected_command = response[command_start:command_end]
                    # 不再继续处理后续内容
                    break
        
        print("\n" + "-"*40)
        
        if detected_command:
            print(f"检测到指令: {detected_command}")
            # 这里可以调用对应的函数
            # process_voice_command(detected_command)