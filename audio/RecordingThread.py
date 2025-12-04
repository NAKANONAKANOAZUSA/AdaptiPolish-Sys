import time
import wave
import tempfile
import numpy as np
import sounddevice as sd
from PyQt5.QtCore import QThread, pyqtSignal


class RecordingThread(QThread):
    """后台录音线程，使用预校准的阈值"""
    recording_finished = pyqtSignal(str)  # 录音完成信号，传递文件路径
    status_updated = pyqtSignal(str)  # 状态更新信号

    def __init__(self, threshold, silence_duration=2.0, max_duration=30.0):
        super().__init__()
        self.stop_recording = False
        self.fs = 48000  # 采样率
        self.channels = 1
        self.dtype = 'int16'
        self.frames = []
        self.max_duration = max_duration  # 最大录音时长（秒）
        self.silence_duration = silence_duration  # 静音时长阈值（秒）
        self.silence_start_time = None  # 静音开始时间
        self.start_time = None
        self.is_silent = False
        self.threshold = threshold  # 使用预校准的阈值

    def run(self):
        """线程主函数"""
        try:
            # 创建临时文件
            temp_wav_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            temp_wav_path = temp_wav_file.name
            temp_wav_file.close()

            # 定义回调函数
            def callback(indata, frame_count, time_info, status):
                if self.stop_recording:
                    return True  # 返回True表示停止录音

                if status:
                    self.status_updated.emit(f"Recording status: {status}")

                # 计算当前音频块的音量（RMS值）
                rms = np.sqrt(np.mean(indata.astype(np.float32) ** 2))

                # 检查是否超过音量阈值
                if rms > self.threshold:
                    # 有声音，重置静音计时
                    self.silence_start_time = None
                    self.is_silent = False

                    # 将数据添加到帧列表
                    self.frames.append(indata.copy())

                    # 检查是否达到最大时长
                    if self.start_time and (time.time() - self.start_time) >= self.max_duration:
                        self.stop_recording = True
                        return True
                else:
                    # 静音状态
                    current_time = time.time()

                    if not self.is_silent:
                        # 第一次检测到静音，记录开始时间
                        self.silence_start_time = current_time
                        self.is_silent = True
                        # self.status_updated.emit("检测到静音...")
                    else:
                        # 检查静音时长是否超过阈值
                        if current_time - self.silence_start_time >= self.silence_duration:
                            self.stop_recording = True
                            self.status_updated.emit("静音时间过长，停止录音")
                            return True

                return False

            # 开始录音
            self.status_updated.emit(f"开始录音 (阈值={self.threshold:.4f})")
            self.start_time = time.time()
            self.silence_start_time = None
            self.is_silent = False

            # 启动输入流
            with sd.InputStream(samplerate=self.fs, channels=self.channels,
                                dtype=self.dtype, callback=callback):
                # 等待停止标志被设置或达到时长限制
                while not self.stop_recording:
                    # 检查是否达到时长限制
                    if self.start_time and (time.time() - self.start_time) >= self.max_duration:
                        break
                    time.sleep(0.1)

            # 保存录音文件
            if self.frames:
                with wave.open(temp_wav_path, 'wb') as wf:
                    wf.setnchannels(self.channels)
                    wf.setsampwidth(2)  # 16-bit audio
                    wf.setframerate(self.fs)
                    wf.writeframes(b''.join(self.frames))

                self.status_updated.emit(f"录音已保存到 {temp_wav_path}")
                self.recording_finished.emit(temp_wav_path)
            else:
                self.status_updated.emit("未检测到有效声音，未保存录音")
                self.recording_finished.emit('')

        except Exception as e:
            self.status_updated.emit(f"录音错误: {str(e)}")
            self.recording_finished.emit('')