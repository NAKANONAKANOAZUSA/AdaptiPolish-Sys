import os
import json
import queue
import tempfile
import threading
import hashlib
from difflib import SequenceMatcher
from pydub import AudioSegment
from pydub.playback import play
class AudioSystem:
    def __init__(self,config, tts_model=None, preload_dir="audio_resources"):
        self.config = config
        self.audio_queue = queue.Queue()  # 音频播放队列
        self.stop_event = threading.Event()  # 停止事件
        self.playing = False  # 当前是否正在播放
        self.preloaded_audio = {}  # 预加载的固定语音
        self.dynamic_cache = {}  # 动态生成语音的缓存
        self.preload_dir = preload_dir  # 预加载目录
        self.preload_complete = False  # 预加载完成标志
        self.tts_model = tts_model  # TTS模型实例
        
        # 创建并启动播放线程
        self.playback_thread = threading.Thread(target=self.play_audio_from_queue, daemon=True)
        self.playback_thread.start()
        
        # 启动预加载线程
        self.preload_thread = threading.Thread(target=self.preload_audio_files, daemon=True)
        self.preload_thread.start()

    def preload_audio_files(self):
        """预加载指定目录中的所有line_*音频文件"""
        print(f"[AUDIO] 开始预加载音频文件: {self.preload_dir}")
        
        try:
            # 确保目录存在
            os.makedirs(self.preload_dir, exist_ok=True)
            
            # 获取所有line_*.wav文件
            audio_files = [f for f in os.listdir(self.preload_dir) 
                        if f.startswith("line_") and f.endswith(".wav")]
            
            # 加载每个音频文件
            for filename in audio_files:
                try:
                    filepath = os.path.join(self.preload_dir, filename)
                    audio = AudioSegment.from_file(filepath)
                    
                    # 生成文本标识（从文件名提取）
                    text_id = filename[5:-4]  # 移除"line_"和".wav"
                    
                    # 保存AudioSegment对象
                    self.preloaded_audio[text_id] = audio
                    
                    print(f"[PRELOAD] 已预加载: {text_id}")
                except Exception as e:
                    print(f"[PRELOAD ERROR] 加载失败: {filename} - {str(e)}")
            
            # 加载预加载映射文件（如果存在）
            mapping_file = os.path.join(self.preload_dir, "audio_mapping.json")
            if os.path.exists(mapping_file):
                try:
                    with open(mapping_file, 'r', encoding='utf-8') as f:
                        raw_mapping = json.load(f)
                    
                    # 标准化映射键：移除首尾空格、转换为小写
                    self.audio_mapping = {
                        key.strip().lower(): value
                        for key, value in raw_mapping.items()
                    }
                    
                    print(f"[PRELOAD] 已加载并标准化音频映射: {len(self.audio_mapping)} 条记录")
                    
                    # 调试：打印映射内容
                    print(f"[DEBUG] 映射文件内容示例:")
                    for i, (key, val) in enumerate(self.audio_mapping.items()):
                        if i < 5:  # 打印前5个条目
                            print(f"  '{key}' => '{val}'")
                    
                except Exception as e:
                    print(f"[PRELOAD ERROR] 加载音频映射失败: {str(e)}")
                    self.audio_mapping = {}
            else:
                self.audio_mapping = {}
                print("[PRELOAD] 未找到音频映射文件")
            
            # 打印预加载音频ID
            print(f"[DEBUG] 预加载音频ID: {list(self.preloaded_audio.keys())}")
            
            self.preload_complete = True
            print(f"[AUDIO] 音频预加载完成! 已加载 {len(self.preloaded_audio)} 个文件")
        except Exception as e:
            print(f"[PRELOAD ERROR] 预加载失败: {str(e)}")
            self.preload_complete = False

    def generate_audio(self, text):
        """使用TTS生成音频并返回AudioSegment对象"""
        try:
            # 创建临时文件
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            temp_path = temp_file.name
            temp_file.close()
            
            # 使用TTS生成语音
            self.tts_model.run(text, output_path=temp_path)
            
            # 加载生成的音频
            audio = AudioSegment.from_file(temp_path)
            
            # 删除临时文件
            try:
                os.remove(temp_path)
            except:
                pass
            
            return audio
        except Exception as e:
            print(f"[TTS ERROR] 生成音频失败: {str(e)}")
            # 返回静音作为回退
            return AudioSegment.silent(duration=1000)

    def get_audio_hash(self, text):
        """生成文本的哈希值作为唯一标识"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()[:10]

    def add_to_queue(self, audio_text):
        """将音频文本添加到播放队列"""
        if not audio_text:
            return
        
        # 调试日志
        print(f"[AUDIO] 请求音频: '{audio_text}'")
        print(f"[DEBUG] 当前映射键: {list(self.audio_mapping.keys())[:5]}...")
        
        # 标准化文本（小写+去除首尾空格）
        normalized_text = audio_text.strip().lower()
        print(f"[DEBUG] 标准化文本: '{normalized_text}'")
        
        # 1. 检查文本是否在预加载映射中（精确匹配）
        if normalized_text in self.audio_mapping:
            mapped_id = self.audio_mapping[normalized_text]
            print(f"[AUDIO] 精确匹配: '{audio_text}' -> '{mapped_id}'")
            
            # 检查文件名格式并提取ID
            if mapped_id.startswith("line_") and mapped_id.endswith(".wav"):
                text_id = mapped_id[5:-4]  # 提取数字ID
                
                # 检查预加载缓存
                if text_id in self.preloaded_audio:
                    self.audio_queue.put(("preloaded", text_id))
                    print(f"[AUDIO] 使用预加载音频: ID={text_id}")
                    return
                else:
                    print(f"[WARNING] 映射ID未预加载: {text_id}")
        
        # 2. 模糊匹配：检查文本是否包含映射键（或映射键包含文本）
        for key, mapped_id in self.audio_mapping.items():
            # 检查文本是否包含键或键包含文本
            if key in normalized_text or normalized_text in key:
                print(f"[AUDIO] 模糊匹配: '{audio_text}' ~ '{key}' -> '{mapped_id}'")
                
                # 检查文件名格式并提取ID
                if mapped_id.startswith("line_") and mapped_id.endswith(".wav"):
                    text_id = mapped_id[5:-4]  # 提取数字ID
                    
                    # 检查预加载缓存
                    if text_id in self.preloaded_audio:
                        self.audio_queue.put(("preloaded", text_id))
                        print(f"[AUDIO] 使用预加载音频: ID={text_id}")
                        return
                    else:
                        print(f"[WARNING] 模糊匹配ID未预加载: {text_id}")
        
        # 3. 检查文本是否在预加载文件中（精确匹配）
        if normalized_text in self.preloaded_audio:
            self.audio_queue.put(("preloaded", normalized_text))
            print(f"[AUDIO] 使用预加载音频: {normalized_text}")
            return
        
        # 4. 检查动态缓存
        audio_hash = self.get_audio_hash(audio_text)
        if audio_hash in self.dynamic_cache:
            self.audio_queue.put(("dynamic", audio_hash))
            print(f"[AUDIO] 使用缓存动态音频: {audio_text}")
            return
        
        # 5. 需要动态生成
        print(f"[AUDIO] 将动态生成音频: {audio_text}")
        self.audio_queue.put(("generate", audio_text))
        
        # 6. 记录未匹配的文本（用于调试）
        closest_keys = []
        for key in self.audio_mapping.keys():
            # 计算相似度
            similarity = SequenceMatcher(None, normalized_text, key).ratio()
            closest_keys.append((key, similarity))
        
        # 按相似度排序
        closest_keys.sort(key=lambda x: x[1], reverse=True)
        
        print(f"[WARNING] 未找到匹配的预加载音频: '{audio_text}'")
        print(f"[DEBUG] 最接近的5个键:")
        for key, similarity in closest_keys[:5]:
            print(f"  '{key}' (相似度: {similarity:.2f})")

    def play_audio_from_queue(self):
        """播放队列中的音频"""
        while not self.stop_event.is_set():
            try:
                item = self.audio_queue.get(timeout=1.0)
                item_type, identifier = item
                
                if item_type == "preloaded":
                    self.play_preloaded_audio(identifier)
                elif item_type == "dynamic":
                    self.play_dynamic_audio(identifier)
                else:  # generate
                    self.generate_and_play_audio(identifier)
                
                self.audio_queue.task_done()
            except queue.Empty:
                pass

    def play_preloaded_audio(self, audio_id):
        """播放预加载的音频"""
        try:
            if audio_id not in self.preloaded_audio:
                print(f"[ERROR] 预加载音频不存在: {audio_id}")
                return
            
            # 获取预加载的音频
            audio_segment = self.preloaded_audio[audio_id]
            
            # 提高音量
            louder_audio = audio_segment + 10  # 增加10dB
            
            print(f"[AUDIO] 开始播放: {audio_id}")
            
            # 播放音频
            play(louder_audio)
            
            print(f"[AUDIO] 播放完成: {audio_id}")
        except Exception as e:
            print(f"[PLAY ERROR] 播放失败: {str(e)}")

    def generate_and_play_audio(self, text):
        """动态生成并播放音频"""
        try:
            print(f"[TTS] 生成音频: {text}")
            
            # 生成音频
            audio_segment = self.generate_audio(text)
            audio_hash = self.get_audio_hash(text)
            
            # 缓存音频数据
            self.dynamic_cache[audio_hash] = audio_segment
            
            # 播放音频
            self.play_dynamic_audio(audio_hash)
            
            print(f"[AUDIO] 播放完成: {text}")
        except Exception as e:
            print(f"[PLAY ERROR] 播放失败: {str(e)}")

    def play_dynamic_audio(self, audio_hash):
        """播放动态生成的音频"""
        try:
            if audio_hash not in self.dynamic_cache:
                print(f"[ERROR] 动态音频不存在: {audio_hash}")
                return
            
            # 获取缓存的音频
            audio_segment = self.dynamic_cache[audio_hash]
            
            # 提高音量
            louder_audio = audio_segment + 10  # 增加10dB
            
            print(f"[AUDIO] 开始播放: {audio_hash}")
            
            # 播放音频
            play(louder_audio)
            
            print(f"[AUDIO] 播放完成: {audio_hash}")
        except Exception as e:
            print(f"[PLAY ERROR] 播放失败: {str(e)}")

    def stop(self):
        """停止音频系统"""
        self.stop_event.set()
        
        # 等待播放线程结束
        if self.playback_thread.is_alive():
            self.playback_thread.join(timeout=2.0)
        
        # 清空队列
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break
        
        print("[AUDIO] 音频系统已停止")

    def is_playing(self):
        """检查是否正在播放音频"""
        return self.playing