import time
import numpy as np
import sounddevice as sd
from PyQt5.QtCore import QThread, pyqtSignal



class MicrophoneCalibrator(QThread):
    """麦克风校准线程"""
    calibration_done = pyqtSignal(float, float, float)  # 信号：校准完成 (背景音量, 说话音量, 阈值)
    status_updated = pyqtSignal(str)  # 信号：状态更新

    def __init__(self):
        super().__init__()
        self.stop_calibration = False
        self.fs = 48000  # 采样率
        self.channels = 1
        self.dtype = 'int16'
        self.background_level = None
        self.speech_level = None
        self.threshold = None

    def run(self):
        """执行校准过程"""
        try:
            # 背景噪音校准
            self.status_updated.emit("正在进行背景噪音校准... (请保持安静)")
            self.calibrate_background_level()

            if self.stop_calibration:
                self.status_updated.emit("校准被取消")
                return

            # 说话音量校准
            self.status_updated.emit("正在进行说话音量校准... (请说几句话)")
            self.calibrate_speech_level()

            if self.stop_calibration:
                self.status_updated.emit("校准被取消")
                return

            # 计算阈值
            self.calculate_threshold()
            self.status_updated.emit(
                f"校准完成: 背景噪音={self.background_level:.4f}, 说话音量={self.speech_level:.4f}, 阈值={self.threshold:.4f}")
            self.calibration_done.emit(self.background_level, self.speech_level, self.threshold)

        except Exception as e:
            self.status_updated.emit(f"校准出错: {str(e)}")

    def calibrate_background_level(self, calibration_time=2.0):
        """校准背景噪音水平"""
        background_rms = []
        start_time = time.time()

        def background_callback(indata, frame_count, time_info, status):
            if self.stop_calibration:
                return True

            if status:
                self.status_updated.emit(f"背景校准状态: {status}")

            # 计算当前块的RMS
            rms = np.sqrt(np.mean(indata.astype(np.float32) ** 2))
            background_rms.append(rms)

            # 检查是否达到校准时间
            if time.time() - start_time >= calibration_time:
                return True

            return False

        # 开始背景噪音校准
        with sd.InputStream(samplerate=self.fs, channels=self.channels,
                            dtype=self.dtype, callback=background_callback):
            while time.time() - start_time < calibration_time and not self.stop_calibration:
                time.sleep(0.1)

        if background_rms:
            # 计算背景噪音的平均RMS值
            self.background_level = np.mean(background_rms)
            self.status_updated.emit(f"背景噪音校准完成: {self.background_level:.4f}")
        else:
            self.background_level = 0.01  # 默认值
            self.status_updated.emit("背景噪音校准失败，使用默认值")

    def calibrate_speech_level(self, calibration_time=3.0):
        """校准正常说话音量水平"""
        speech_rms = []
        start_time = time.time()
        max_speech = 0.0
        min_speech = 1.0

        def speech_callback(indata, frame_count, time_info, status):
            nonlocal max_speech, min_speech  # 添加这行声明

            if self.stop_calibration:
                return True

            if status:
                self.status_updated.emit(f"说话校准状态: {status}")

            # 计算当前块的RMS
            rms = np.sqrt(np.mean(indata.astype(np.float32) ** 2))

            # 只考虑明显高于背景噪音的块
            if rms > (self.background_level or 0.01) * 1.5:
                speech_rms.append(rms)
                if rms > max_speech:
                    max_speech = rms
                if rms < min_speech:
                    min_speech = rms

            # 检查是否达到校准时间
            if time.time() - start_time >= calibration_time:
                return True

            return False

        # 开始说话音量校准
        with sd.InputStream(samplerate=self.fs, channels=self.channels,
                            dtype=self.dtype, callback=speech_callback):
            while time.time() - start_time < calibration_time and not self.stop_calibration:
                time.sleep(0.1)

        if speech_rms:
            # 计算说话音量的平均RMS值，排除过高和过低的异常值
            filtered_rms = [rms for rms in speech_rms if min_speech <= rms <= max_speech]
            if filtered_rms:
                self.speech_level = np.mean(filtered_rms)
            else:
                self.speech_level = np.mean(speech_rms)
            self.status_updated.emit(f"说话音量校准完成: {self.speech_level:.4f}")
        else:
            self.speech_level = (self.background_level or 0.01) * 2  # 默认值
            self.status_updated.emit("说话音量校准失败，使用默认值")

    def calculate_threshold(self):
        """根据校准结果计算阈值"""
        # 计算阈值：背景噪音和说话音量的加权平均值
        if self.background_level > 0 and self.speech_level > self.background_level:
            # 使用背景噪音和说话音量的中间值作为阈值
            self.threshold = self.background_level * 0.7 + self.speech_level * 0.3

            # 确保阈值至少比背景噪音高50%
            min_threshold = self.background_level * 1.5
            if self.threshold < min_threshold:
                self.threshold = min_threshold
        else:
            # 校准失败时使用默认阈值
            self.threshold = 0.02

        self.status_updated.emit(f"计算完成: 阈值={self.threshold:.4f}")