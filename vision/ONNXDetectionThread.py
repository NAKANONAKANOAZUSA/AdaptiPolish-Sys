import os
import sys
import time
import threading
import numpy as np
import cv2
from datetime import datetime


# ONNX运行时
try:
    import onnxruntime as ort
except ImportError:
    print("警告: 未找到onnxruntime，ONNX功能将不可用")

# PyQt5相关
from PyQt5.QtCore import QThread, pyqtSignal
from vision.CameraDetectionSystem import CameraDetectionSystem
from vision.contour_manager import FixedContourManager

class ONNXDetectionThread(QThread):
    """模型识别，自动标定"""
    update_frame = pyqtSignal(np.ndarray)  # 信号：更新视频帧
    detection_result = pyqtSignal(str)  # 信号：发送检测结果文本
    detection_coords = pyqtSignal(list)  # 信号：发送检测到的坐标列表
    single_image_result = pyqtSignal(np.ndarray, str)  # 信号：单张图像检测结果
    calibration_status = pyqtSignal(str)  # 信号：发送标定状态信息

    def __init__(self,ip='0.0.0.0', port=9999,model_path=None,config = None):
        super().__init__()
        self.config = config
        self.model_path = model_path  # ONNX模型文件路径
        self.ip = ip  # 摄像头IP地址
        self.port = port  # 摄像头端口号
        # 创建摄像头检测系统实例
        self.detection_system = CameraDetectionSystem(config=config,server_ip=ip, server_port=port)
        self.active = False  # 线程活动状态标志
        self.cap = None  # OpenCV视频捕获对象
        self.camera_type = "local"  # 摄像头类型（local/network）
        self.single_image_path = None  # 单张图像检测路径
        self.model_shape = (640, 640)  # 模型输入尺寸
        self.conf_threshold = 0.25  # 目标检测置信度阈值
        self.num_classes = None  # 模型类别数量
        self.total_frames = 0  # 处理的总帧数
        self.prev_frame_time = 0  # 上一帧处理时间
        self.start_time = time.time()  # 线程启动时间
        self.output_dir = "contour_data"  # 轮廓数据输出目录
        self.calibration_complete = False  # 标定状态

    def load_model(self):
        """加载模型并初始化检测系统"""
        try:
            # 确保检测系统正确加载模型
            if not self.detection_system.load_model(self.model_path):
                return False

            # 获取模型输入尺寸
            input_info = self.detection_system.session.get_inputs()[0]
            input_shape = input_info.shape
            if len(input_shape) == 4:
                height = input_shape[2]
                width = input_shape[3]
                self.model_shape = (width, height)
            elif len(input_shape) == 3:
                height = input_shape[1]
                width = input_shape[2]
                self.model_shape = (width, height)
            else:
                self.model_shape = (640, 640)

            # 动态检测类别数量
            output_info = self.detection_system.session.get_outputs()[0]
            output_shape = output_info.shape
            if len(output_shape) >= 2:
                self.num_classes = output_shape[1] - 4
            else:
                self.num_classes = 80
            self.detection_system.load_calibration_params()
            return True
        except Exception as e:
            print(f"加载模型失败: {str(e)}")
            return False

    def run(self):
        try:
            if not self.load_model():
                self.detection_result.emit("ONNX模型加载失败")
                return

            # 单张图像处理流程
            if self.single_image_path:
                self.process_single_image(self.single_image_path)
                return

            self.active = True

            # 只有在网络摄像头模式下才启动接收
            if self.camera_type == "network":
                self.detection_system.start_receiving()

            # 视频流处理主循环
            self.process_video_stream()
        except Exception as e:
            print(f"检测线程异常: {str(e)}")
            import traceback
            traceback.print_exc()
        finally:
            self.cleanup_resources()
            self.active = False

    def set_camera_type(self, camera_type):
        """设置摄像头类型"""
        self.camera_type = camera_type

    def cleanup_resources(self):
        """清理线程资源"""
        print("[THREAD] 清理资源")
        # 关闭检测系统
        if hasattr(self.detection_system, 'stop'):
            self.detection_system.stop_event.set()
            if self.detection_system.receive_thread and self.detection_system.receive_thread.is_alive():
                self.detection_system.receive_thread.join(timeout=0.5)

        # 关闭本地摄像头
        if self.cap and self.cap.isOpened():
            self.cap.release()
            print("[THREAD] 本地摄像头已释放")

        # 关闭OpenCV窗口
        cv2.destroyAllWindows()

    def perform_calibration(self):
        """执行标定流程"""
        if not hasattr(self, 'detection_system') or self.detection_system is None:
            print("[ERROR] 检测系统未初始化")
            self.calibration_status.emit("标定失败：检测系统未初始化")
            return
        if self.detection_system.calibration_complete:
            self.detection_system.calibration_complete = False

        # 启动标定线程
        threading.Thread(target=self._calibration_task, daemon=True).start()
        self.calibration_status.emit("标定已启动...")

    def _calibration_task(self):
        """后台标定任务"""
        print("[CALIBRATION] 开始自动标定")
        self.calibration_status.emit("开始标定，请确保ArUco标记在视野中")

        # 重置状态
        self.detection_system.calibration_complete = False
        self.detection_system.calibration_in_progress = True
        self.detection_system.calibration_frames = []

        # 收集标定帧
        collected_frames = 0
        max_frames = 30
        timeout = time.time() + 30

        while collected_frames < max_frames and time.time() < timeout and self.active:
            try:
                frame = self.capture_frame()
                print(frame)
                if frame is None:
                    print("[CALIBRATION] 未获取到帧")
                    time.sleep(0.1)
                    continue

                # 处理ArUco标记
                result = self.detection_system.detect_aruco_markers(frame)
                if result:
                    x1, x2, y1, y2 = result
                    self.detection_system.calibration_frames.append((x1, x2, y1, y2))
                    collected_frames += 1

                    # 发送标定状态
                    self.calibration_status.emit(f"采集帧: {collected_frames}/{max_frames}")

                    # 在图像上标记
                    frame_copy = frame.copy()
                    cv2.putText(frame_copy, f"Calibration: {collected_frames}/{max_frames}",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                    # 显示标记位置
                    cv2.circle(frame_copy, (x1, y1), 10, (0, 255, 0), 2)
                    cv2.circle(frame_copy, (x2, y2), 10, (0, 255, 0), 2)

                    # 发射更新信号
                    self.update_frame.emit(frame_copy)
                else:
                    # 显示提示信息
                    frame_copy = frame.copy()
                    cv2.putText(frame_copy, "未检测到ArUco标记", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    self.update_frame.emit(frame_copy)

            except Exception as e:
                print(f"[CALIBRATION ERROR] {str(e)}")
                import traceback
                traceback.print_exc()

            time.sleep(0.05)

        # 计算标定参数
        if len(self.detection_system.calibration_frames) >= 10:
            # 计算平均值
            sum_x1 = sum([x1 for x1, _, _, _ in self.detection_system.calibration_frames])
            sum_x2 = sum([x2 for _, x2, _, _ in self.detection_system.calibration_frames])
            sum_y1 = sum([y1 for _, _, y1, _ in self.detection_system.calibration_frames])
            sum_y2 = sum([y2 for _, _, _, y2 in self.detection_system.calibration_frames])

            self.detection_system.c_x = (sum_x1 + sum_x2) / (len(self.detection_system.calibration_frames) * 2.0)
            self.detection_system.c_y = (sum_y1 + sum_y2) / (len(self.detection_system.calibration_frames) * 2.0)

            avg_distance = 0
            for x1, x2, y1, y2 in self.detection_system.calibration_frames:
                avg_distance += ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
            avg_distance /= len(self.detection_system.calibration_frames)

            # 实际距离为220mm
            if avg_distance > 0:
                self.detection_system.ratio = 220.0 / avg_distance
            else:
                self.detection_system.ratio = 1.0

            self.detection_system.calibration_complete = True

            # 标定结果
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            calibration_data = f"{timestamp}: c_x={self.detection_system.c_x:.2f}, c_y={self.detection_system.c_y:.2f}, ratio={self.detection_system.ratio:.4f}\n"
            with open("calibration_log.txt", "a") as f:
                f.write(calibration_data)

            self.detection_system.save_calibration_params(self,)

            print(f"[CALIBRATION] 标定完成! {calibration_data.strip()}")
            self.calibration_status.emit("标定成功!")
        else:
            print("[CALIBRATION] 标定失败: 采集帧不足")
            self.calibration_status.emit("标定失败: 标记未检测到")

        self.detection_system.calibration_in_progress = False

    def capture_frame(self):
        """从适当源捕获帧"""
        try:
            if self.camera_type == "network":
                # 增加重试机制
                for _ in range(10):  # 增加重试次数
                    if not self.detection_system.frame_queue.empty():
                        frame = self.detection_system.frame_queue.get()
                        return frame
                    time.sleep(0.05)
                return None
            else:  # 本地摄像头
                with threading.Lock():
                    if self.cap is None:
                        # 根据操作系统选择不同的后端
                        if sys.platform.startswith('linux'):
                            # Linux系统使用V4L2
                            self.cap = cv2.VideoCapture(20, cv2.CAP_V4L2)
                        elif sys.platform.startswith('win'):
                            # Windows系统使用DShow
                            self.cap = cv2.VideoCapture(20, cv2.CAP_DSHOW)
                        else:
                            # 其他系统使用默认后端
                            self.cap = cv2.VideoCapture(20)

                        if not self.cap.isOpened():
                            print("[ERROR] 无法打开本地摄像头")
                            return None

                ret, frame = self.cap.read()
                return frame
        except Exception as e:
            print(f"捕获帧错误: {str(e)}")
            return None

    def process_video_stream(self):
        """视频流处理主循环"""
        frame_count = 0

        # 初始化固定轮廓管理器
        contour_manager = FixedContourManager(config=self.config)
        fixed_contour = None
        fixed_contour_size = None

        # 确保输出目录存在
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"确保输出目录已创建: {self.output_dir}")

        while self.active:
            frame = self.capture_frame()
            if frame is None:
                # 显示等待信息
                blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(blank_frame, "等待视频流...", (100, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                self.update_frame.emit(blank_frame)
                time.sleep(0.1)
                continue

            # 处理当前帧
            processed_frame = frame.copy()
            frame_count += 1

            # 如果已经计算好固定轮廓
            if contour_manager.completed and fixed_contour:
                # 只绘制固定轮廓
                processed_frame = self.detection_system.draw_fixed_contour(processed_frame, fixed_contour,
                                                                           fixed_contour_size)

                # 只保存一次固定轮廓
                if frame_count == contour_manager.max_frames + 1:
                    self.detection_system.save_contour_data(contour_data, frame_count)
            else:
                # 如果正在标定，跳过常规检测
                if self.detection_system.calibration_in_progress:
                    time.sleep(0.05)
                    continue

                # 显示ArUco标记
                if not self.detection_system.calibration_complete:
                    result = self.detection_system.detect_aruco_markers(processed_frame)
                    if result:
                        x1, x2, y1, y2 = result
                        # 在图像上显示标记位置
                        cv2.circle(processed_frame, (x1, y1), 10, (0, 255, 0), 2)
                        cv2.circle(processed_frame, (x2, y2), 10, (0, 255, 0), 2)

                # 执行目标检测
                if self.detection_system.session and self.detection_system.calibration_complete:
                    try:
                        processed_frame, detections, world_coords = self.detection_system.detect_objects(
                            processed_frame)

                        # 准备检测结果
                        message = "未检测到目标"
                        contour_data = None

                        if detections:
                            objects = [f"{det['label']}:{det['score']:.2f}" for det in detections]
                            message = ', '.join(objects)

                            # 获取轮廓数据
                            if 'contour' in detections[0]:
                                contour_data = detections[0]['contour']

                                # 添加到轮廓管理器
                                contour_manager.add_contour(contour_data)

                                # 如果管理器完成处理，获取固定轮廓
                                if contour_manager.completed and not fixed_contour:
                                    fixed_contour, fixed_contour_size = contour_manager.get_fixed_contour()
                                    print(f"\n固定轮廓已生成 | 点数: {len(fixed_contour)}")
                                    self.detection_system.fixed_contour = fixed_contour
                                    fixed_contour_size = contour_manager.smoothed_size
                                    # 保存到检测系统
                                    self.detection_system.fixed_contour = fixed_contour
                                    self.detection_system.fixed_contour_size = fixed_contour_size

                                # 保存轮廓数据
                                self.detection_system.save_contour_data(contour_data, frame_count)
                                print(
                                    f"\n检测到目标 | 类别: {contour_data['class_id']} | 置信度: {contour_data['confidence']:.2f}")

                        # 发射检测信号
                        self.detection_result.emit(message)
                        self.detection_coords.emit(world_coords)
                    except Exception as e:
                        print(f"目标检测错误: {str(e)}")
                        import traceback
                        traceback.print_exc()

            # 显示标定状态
            status_text = "Calibration: " + (
                "COMPLETE" if self.detection_system.calibration_complete
                else "IN PROGRESS" if self.detection_system.calibration_in_progress
                else "INCOMPLETE"
            )
            status_color = (0, 255, 0) if self.detection_system.calibration_complete else (0, 0, 255)
            cv2.putText(processed_frame, status_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

            # 显示转换参数
            if self.detection_system.calibration_complete:
                params_text = f"c_x:{self.detection_system.c_x:.1f} c_y:{self.detection_system.c_y:.1f} r:{self.detection_system.ratio:.4f}"
                cv2.putText(processed_frame, params_text, (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            status_text = "Fixed" if fixed_contour else f"Collecting: {len(contour_manager.raw_contours)}/{contour_manager.max_frames}"

            # 绘制叠加信息
            cv2.putText(processed_frame, f"Frame: {frame_count}",
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(processed_frame, f"Status: {status_text}",
                        (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # 点数量显示
            if contour_manager.raw_contours:
                points_count = len(contour_manager.raw_contours[-1]['contour_points'])
                cv2.putText(processed_frame, f"Points: {points_count}",
                            (10, processed_frame.shape[0] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # 发射更新帧信号
            self.update_frame.emit(processed_frame)

            time.sleep(0.01)

    def process_single_image(self, img_path):
        """处理单张图像检测"""
        try:
            frame = cv2.imread(img_path)
            if frame is None:
                raise ValueError(f"无法读取图像: {img_path}")

            # 临时的轮廓管理器
            contour_manager = FixedContourManager(max_frames=1)
            fixed_contour = None
            fixed_contour_size = None

            # 确保检测系统已加载模型
            if not self.detection_system.session:
                self.load_model()

            # 处理当前帧
            orig_h, orig_w = frame.shape[:2]
            img_size = (orig_w, orig_h)

            # 使用新的推理流程
            boxes, masks, protos, _ = self.detection_system.infer(frame)
            contour_data, processed_frame = self.detection_system.extract_contour_data(frame, boxes, masks, protos,
                                                                                       img_size)

            # 添加到轮廓管理器
            if contour_data:
                contour_manager.add_contour(contour_data)
                print(f"检测到目标 | 类别: {contour_data['class_id']} | 置信度: {contour_data['confidence']:.2f}")

                # 保存轮廓数据
                frame_count = 0  # 单帧处理使用0作为帧号
                self.save_contour_data(contour_data, frame_count, False)

                # 如果管理器完成处理，获取固定轮廓
                if contour_manager.completed and not fixed_contour:
                    fixed_contour, fixed_contour_size = contour_manager.get_fixed_contour()
                    print(f"\n生成固定轮廓 | 点数: {len(fixed_contour)}")

                    # 绘制固定轮廓
                    processed_frame = self.draw_fixed_contour(processed_frame, fixed_contour, img_size)

                    # 保存固定轮廓数据
                    fixed_contour_data = {
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "confidence": 1.0,
                        "class_id": contour_manager.tracking_id or -1,
                        "bbox": [0, 0, img_size[0], img_size[1]],
                        "contour_points": fixed_contour,
                        "image_size": img_size,
                        "points_count": len(fixed_contour)
                    }
                    self.save_contour_data(fixed_contour_data, frame_count, True)

            # 准备结果文本
            result_text = "未检测到目标"
            if contour_data:
                objects = [f"{contour_data['class_id']}:{contour_data['confidence']:.2f}"]
                result_text = ', '.join(objects)

                # 添加点数量信息
                result_text += f" | 轮廓点: {len(contour_data['contour_points'])}"

            # 添加状态信息到图像
            cv2.putText(processed_frame, f"图像: {os.path.basename(img_path)}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(processed_frame, f"结果: {result_text}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # 显示点数量
            if contour_data:
                points_count = len(contour_data['contour_points'])
                cv2.putText(processed_frame, f"Points: {points_count}",
                            (10, processed_frame.shape[0] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # 发射结果信号
            self.single_image_result.emit(processed_frame, result_text)

        except Exception as e:
            error_msg = f"处理图像时出错: {str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            self.single_image_result.emit(None, error_msg)

    def extract_contour_data(self, image, boxes, masks, protos, img_size):
        """单图像轮廓提取"""
        # 如果没有检测到目标，直接返回
        if not boxes:
            return None, image

        # 获取目标边界框信息
        x, y, w, h = boxes[0]['bbox']
        bbox_x1 = int(x - w / 2)
        bbox_y1 = int(y - h / 2)
        bbox_x2 = int(x + w / 2)
        bbox_y2 = int(y + h / 2)

        # 处理掩膜
        masks_np = np.stack(masks, axis=0)
        protos_flat = protos.reshape(protos.shape[0], -1)
        mask_output = masks_np @ protos_flat
        mask_output = 1 / (1 + np.exp(-mask_output))
        mask_output = mask_output.reshape(-1, protos.shape[1], protos.shape[2])
        m = mask_output[0]
        m = cv2.resize(m, img_size, interpolation=cv2.INTER_LINEAR)

        # 二值化掩膜
        _, binary_mask = cv2.threshold(m, 0.5, 255, cv2.THRESH_BINARY)
        binary_mask = binary_mask.astype(np.uint8)

        # 查找主轮廓
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, image

        # 选择最大的轮廓
        main_contour = max(contours, key=cv2.contourArea)

        # 简化轮廓
        epsilon = 0.005 * cv2.arcLength(main_contour, True)
        simplified = cv2.approxPolyDP(main_contour, epsilon, True)

        # 转换为点列表
        contour_points = [point[0].tolist() for point in simplified]

        # 创建轮廓数据结构
        contour_data = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "confidence": boxes[0]['confidence'],
            "class_id": boxes[0]['class_id'],
            "bbox": boxes[0]['bbox'],
            "contour_points": contour_points,
            "image_size": img_size
        }

        # 准备轮廓点用于绘制
        contour_array = np.array(contour_points, dtype=np.int32).reshape((-1, 1, 2))

        # 绘制轮廓
        overlay = image.copy()
        cv2.rectangle(overlay, (bbox_x1, bbox_y1), (bbox_x2, bbox_y2), (0, 255, 0), 2)
        cv2.drawContours(overlay, [contour_array], -1, (0, 0, 255), 3)

        # 添加点数量显示
        cv2.putText(overlay, f"Points: {len(contour_points)}",
                    (10, image.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return contour_data, overlay

    def set_single_image(self, image_path):
        """设置单张图像检测路径"""
        self.single_image_path = image_path

    def calibrate_single_image(self, img_path):
        """使用单张图像进行标定"""
        try:
            frame = cv2.imread(img_path)
            if frame is None:
                raise ValueError(f"无法读取图像: {img_path}")

            self.calibration_status.emit("开始标定单张图像")

            # 重置标定状态
            self.detection_system.calibration_complete = False
            self.detection_system.calibration_in_progress = True
            self.detection_system.calibration_frames = []

            # 处理图像
            processed_frame = frame.copy()
            result = self.detection_system.detect_aruco_markers(processed_frame)

            if not result:
                self.calibration_status.emit("标定失败: 未检测到ArUco标记")
                return False

            x1, x2, y1, y2 = result
            self.detection_system.calibration_frames.append((x1, x2, y1, y2))

            # 计算标定参数
            self.detection_system.c_x = (x1 + x2) / 2.0
            self.detection_system.c_y = (y1 + y2) / 2.0

            # 计算距离
            distance = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
            if distance > 0:
                self.detection_system.ratio = 220.0 / distance
            else:
                self.detection_system.ratio = 1.0

            self.detection_system.calibration_complete = True

            # 记录标定结果
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            calibration_data = f"{timestamp}: c_x={self.detection_system.c_x:.2f}, c_y={self.detection_system.c_y:.2f}, ratio={self.detection_system.ratio:.4f}\n"
            with open("calibration_log.txt", "a") as f:
                f.write(calibration_data)

            print(f"[CALIBRATION] 标定完成! {calibration_data.strip()}")
            self.calibration_status.emit("标定成功!")

            # 在图像上显示标定结果
            cv2.putText(processed_frame, "Calibration: COMPLETE",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            params_text = f"c_x:{self.detection_system.c_x:.1f} c_y:{self.detection_system.c_y:.1f} r:{self.detection_system.ratio:.4f}"
            cv2.putText(processed_frame, params_text, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            # 发射更新帧信号
            self.update_frame.emit(processed_frame)
            # 标定完成后重置检测状态
            self.detection_system.reset_for_detection()

            return True

        except Exception as e:
            print(f"[CALIBRATION ERROR] {str(e)}")
            self.calibration_status.emit(f"标定失败: {str(e)}")
            return False

    def stop(self):
        """线程安全停止"""
        self.active = False
        self.cleanup_resources()
        self.wait(2000)