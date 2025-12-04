import requests
import json
import re

llm_model_path = 'deepseek-r1-1.5b'

class LlmModel:
    def __init__(self, base_url="http://127.0.0.1:11434", model_path=llm_model_path, system_prompt=None):
        self.base_url = base_url
        self.model = model_path
        
        # 系统提示语（包含机器人控制指令模板）
        self.system_prompt = """
        **角色设定：**
        你是机器人控制系统的智能助手，负责处理机器人操作指令和技术问题。
        
        **指令识别：**
        当用户发出操作指令时，请返回对应的指令代码：
        
        **核心机器人控制指令：**
        - 启动/停止摄像头: TOGGLE_CAMERA
        - 连接/断开机器人: CONNECT_ROBOT / DISCONNECT_ROBOT
        - 开始标定: START_CALIBRATION
        - 启动/停止打磨: START_GRINDING / STOP_GRINDING
        - 启动/停止语音识别: START_SPEECH_RECOGNITION / STOP_SPEECH_RECOGNITION
        
        **示教点管理指令：**
        - 保存当前位置: SAVE_TEACH_POINT
        - 移动到指定点: MOVE_TO_POINT:[点名称]
        - 删除示教点: DELETE_POINT:[点名称]
        - 执行所有点: EXECUTE_ALL_POINTS
        
        **回答风格：**
        1. 用中文回答，简洁明了
        2. 直接回答问题，避免无关内容
        3. 重点突出，优先回答核心问题
        4. 总回复长度不超过40字
        """
        
        # 预设回答模板
        self.response_templates = {
            "TOGGLE_CAMERA": "正在切换摄像头状态...",
            "CONNECT_ROBOT": "正在连接机器人...",
            "DISCONNECT_ROBOT": "正在断开机器人连接...",
            "START_CALIBRATION": "开始标定流程...",
            "START_GRINDING": "正在启动打磨操作...",
            "STOP_GRINDING": "正在停止打磨操作...",
            "START_SPEECH_RECOGNITION": "启动语音识别功能...",
            "STOP_SPEECH_RECOGNITION": "停止语音识别功能...",
            "SAVE_TEACH_POINT": "已保存当前位置为示教点",
            "MOVE_TO_POINT": "正在移动到指定点...",
            "DELETE_POINT": "已删除指定示教点",
            "EXECUTE_ALL_POINTS": "开始执行所有示教点..."
        }
    
    def _check_preset_responses(self, text):
        """检查预设响应"""
        text_lower = text.lower().strip()
        
        # 简化的指令映射表
        command_map = {
            ("摄像头", "摄像", "相机", "视频"): "TOGGLE_CAMERA",
            ("连接机器人", "机器人连接", "连接机械臂"): "CONNECT_ROBOT",
            ("断开机器人", "断开连接", "机器人断开"): "DISCONNECT_ROBOT",
            ("标定", "校准", "摄像头标定"): "START_CALIBRATION",
            ("开始打磨", "启动打磨", "运行打磨"): "START_GRINDING",
            ("停止打磨", "结束打磨", "暂停打磨"): "STOP_GRINDING",
            ("启动语音识别", "开启语音控制"): "START_SPEECH_RECOGNITION",
            ("停止语音识别", "关闭语音控制"): "STOP_SPEECH_RECOGNITION",
            ("保存点", "记录位置", "保存位置"): "SAVE_TEACH_POINT",
            ("移动到点", "运行到点", "执行点"): "MOVE_TO_POINT",
            ("删除点", "移除点"): "DELETE_POINT",
            ("执行所有点", "运行全部点"): "EXECUTE_ALL_POINTS"
        }
        
        # 检查匹配
        for keywords, command in command_map.items():
            if any(keyword in text_lower for keyword in keywords):
                return self.response_templates.get(command, "指令已接收")
        
        # 如果未匹配任何指令
        return None
    
    def generate(self, text):
        """生成响应流"""
        # 1. 检查预设响应
        preset_response = self._check_preset_responses(text)
        if preset_response:
            yield preset_response
            return
        
        # 2. 准备API请求
        url = f"{self.base_url}/v1/chat/completions"
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": text}
            ],
            "stream": True,
            
            "think": False  # 启用思考过程
            
        }
        
        try:
            # 3. 发送请求并处理流式响应
            with requests.post(url, headers=headers, json=payload, stream=True) as response:
                response.raise_for_status()
                
                full_response = ""
                for line in response.iter_lines():
                    if line:
                        # 解析JSON响应
                        try:
                            data = json.loads(line.decode("utf-8"))
                            content = data["choices"][0]["delta"].get("content", "")
                            full_response += content
                            yield content
                        except (json.JSONDecodeError, KeyError):
                            continue
                
                # 4. 提取指令（如果有）
                command = self._extract_command(full_response)
                if command:
                    yield f"\n\n<command>{command}</command>"
        
        except requests.exceptions.RequestException as e:
            yield f"API调用错误: {str(e)}"
    
    def _extract_command(self, text):
        """从响应文本中提取指令"""
        # 匹配 <command> 标签
        match = re.search(r"<command>(.*?)</command>", text, re.DOTALL)
        return match.group(1).strip() if match else None


# 测试函数 - 放在类定义外部
def test_llm_model():
    # 创建模型实例
    llm = LlmModel()
    
    # 处理用户输入
    def handle_user_input(user_input):
        print(f"用户: {user_input}")
        print("AI: ", end="", flush=True)
        
        for chunk in llm.generate(user_input):
            print(chunk, end="", flush=True)
        
        print("\n" + "-"*40)
    
    # 测试对话
    test_inputs = [
        "请启动打磨操作",
        "我需要停止打磨",
        "现在进行标定",
        "连接机器人",
        "断开麦克风连接",
        "启动语音识别功能",
        "保存当前位置"
    ]

    for input_text in test_inputs:
        handle_user_input(input_text)

# 如果直接运行此文件，则执行测试
if __name__ == "__main__":
    test_llm_model()