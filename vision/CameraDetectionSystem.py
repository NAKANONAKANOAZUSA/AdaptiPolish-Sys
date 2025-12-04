import cv2
import numpy as np
import json
import time
import os
import queue
import threading
import socket
import struct
import onnxruntime as ort


class CameraDetectionSystem:
    """摄像头接收、目标检测、坐标转换的系统"""

    def __init__(self,config,server_ip='0.0.0.0', server_port=9999):
        self.config = config
        #self.model_path = model_path  # ONNX模型文件路径
        self.server_ip = server_ip  # 服务器IP地址（默认监听所有接口）
        self.server_port = server_port  # 服务器端口号
        self.input_size = (640, 640)  # 模型输入尺寸（宽度，高度）
        self.conf_thres = 0.25  # 目标检测置信度阈值（25%以上才保留）
        self.nms_thres = 0.45  # 非极大值抑制阈值（过滤重叠检测框）
        self.num_classes = 80  # 目标类别数量（初始值，实际从模型获取）
        self.calibration_complete = False  # 标定完成标志（摄像头标定状态）
        self.c_x = 0  # 标定中心点X坐标（像素坐标系）
        self.c_y = 0  # 标定中心点Y坐标（像素坐标系）
        self.ratio = 0  # 像素到世界坐标转换比例（毫米/像素）
        self.camera_offset_x = 165  # 相机相对于机器人基座的X偏移(mm)
        self.camera_offset_y = 5  # 相机相对于机器人基座的Y偏移(mm)
        self.calibration_frames = []  # 存储标定过程中收集的帧数据
        self.calibration_in_progress = False  # 标定进行中标志
        self.frame_queue = queue.Queue(maxsize=10)  # 帧队列（存储接收的视频帧）
        self.stop_event = threading.Event()  # 停止事件（用于线程安全停止）
        self.receive_thread = None  # 视频接收线程
        self.session = None  # ONNX运行时会话
        self.input_name = None  # 模型输入节点名称
        self.output_name = None  # 模型输出节点名称
        self.output_dir = "contour_data"  # 轮廓数据输出目录
        self.frame_counter = 0  # 帧计数器（记录处理帧数）
        self.fixed_contour = None  # 存储固定轮廓数据
        self.fixed_contour_size = None  # 存储固定轮廓对应的图像尺寸
        self.save_as_fixed = False  # 保存为固定轮廓标志
        self.init_aruco()  # 初始化ArUco检测器
        self.num_mask = 32  # 掩膜通道数（用于实例分割）
        self.num_classes = 0  # 重置类别数量（实际从模型获取）
        self.image_width = 640  # 默认图像宽度
        self.image_height = 480  # 默认图像高度
        self.model_path = None

    def preprocess(self, image):
        """图像预处理"""
        orig_h, orig_w = image.shape[:2]
        input_tensor = cv2.resize(image, (640, 640))
        input_tensor = cv2.cvtColor(input_tensor, cv2.COLOR_BGR2RGB)
        input_tensor = input_tensor.astype(np.float32) / 255.0
        input_tensor = np.transpose(input_tensor, (2, 0, 1))
        return np.expand_dims(input_tensor, axis=0), (orig_w, orig_h)

    def infer(self, image):
        """推理函数"""
        input_tensor, orig_size = self.preprocess(image)
        outputs = self.session.run(self.output_names, {self.input_name: input_tensor})
        det_output, mask_output = outputs
        predictions = np.squeeze(det_output).T
        mask_protos = mask_output[0]

        boxes, masks = [], []
        orig_w, orig_h = orig_size

        best_confidence = self.config.CONFIDENCE_THRESHOLD
        best_pred = None

        # 循环寻找最高置信度目标
        for pred in predictions:
            if self.num_classes > 0:
                scores = pred[5:5 + self.num_classes]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
            else:
                confidence = pred[4]
                class_id = 0

            # 更新最佳目标
            if confidence > 0.5 and confidence > best_confidence:
                best_confidence = confidence
                best_pred = pred

        # 处理找到的最佳目标
        if best_pred is not None:
            if self.num_classes > 0:
                scores = best_pred[5:5 + self.num_classes]
                class_id = int(np.argmax(scores))
            else:
                class_id = 0

            # 坐标转换和数据处理
            x, y, w, h = best_pred[0:4]
            x *= orig_w / 640
            y *= orig_h / 640
            w *= orig_w / 640
            h *= orig_h / 640
            mask_coeffs = best_pred[-self.num_mask:]

            boxes.append({
                'class_id': class_id,
                'confidence': float(best_confidence),
                'bbox': [float(x), float(y), float(w), float(h)]
            })
            masks.append(mask_coeffs)

        return boxes, masks, mask_protos, (orig_w, orig_h)

    def extract_contour_data(image, boxes, masks, protos, img_size):
        """提取并返回目标的轮廓数据"""
        if not boxes:
            return None, image

        # 获取目标边界框信息
        x, y, w, h = boxes[0]['bbox']
        bbox_x1 = int(x - w / 2)
        bbox_y1 = int(y - h / 2)
        bbox_x2 = int(x + w / 2)
        bbox_y2 = int(y + h / 2)
        bbox_area = w * h

        # 处理掩膜
        masks_np = np.stack(masks, axis=0)
        protos_flat = protos.reshape(32, -1)
        mask_output = masks_np @ protos_flat
        mask_output = 1 / (1 + np.exp(-mask_output))
        mask_output = mask_output.reshape(-1, protos.shape[1], protos.shape[2])
        m = mask_output[0]
        m = cv2.resize(m, img_size, interpolation=cv2.INTER_LINEAR)

        # 二值化掩膜
        _, binary_mask = cv2.threshold(m, 0.5, 255, cv2.THRESH_BINARY)
        binary_mask = binary_mask.astype(np.uint8)
        kernel = np.ones((3, 3), np.uint8)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)

        # 查找所有轮廓
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, image

        # 筛选有效轮廓
        valid_contours = []
        for contour in contours:
            contour_x, contour_y, contour_w, contour_h = cv2.boundingRect(contour)
            contour_x2 = contour_x + contour_w
            contour_y2 = contour_y + contour_h

            overlap_x1 = max(bbox_x1, contour_x)
            overlap_y1 = max(bbox_y1, contour_y)
            overlap_x2 = min(bbox_x2, contour_x2)
            overlap_y2 = min(bbox_y2, contour_y2)

            overlap_width = max(0, overlap_x2 - overlap_x1)
            overlap_height = max(0, overlap_y2 - overlap_y1)
            overlap_area = overlap_width * overlap_height

            contour_area = cv2.contourArea(contour)

            if overlap_area > 0.5 * bbox_area and 0.1 * bbox_area < contour_area < 2.0 * bbox_area:
                valid_contours.append(contour)

        if not valid_contours:
            return None, image

        # 处理主轮廓
        main_contour = max(valid_contours, key=cv2.contourArea)
        contour_points = []

        # 自适应轮廓简化
        if main_contour is not None and len(main_contour) > 4:
            # 计算轮廓周长
            contour_length = cv2.arcLength(main_contour, True)
            base_epsilon = 0.005 * contour_length

            # 计算曲率变化
            curvature_scores = []
            for i in range(1, len(main_contour) - 1):
                p1 = main_contour[i - 1][0]
                p2 = main_contour[i][0]
                p3 = main_contour[i + 1][0]

                # 计算向量
                vec1 = (p1[0] - p2[0], p1[1] - p2[1])
                vec2 = (p3[0] - p2[0], p3[1] - p2[1])

                # 计算角度差
                angle1 = np.arctan2(vec1[1], vec1[0])
                angle2 = np.arctan2(vec2[1], vec2[0])
                angle_diff = np.abs(np.degrees(angle1 - angle2))
                angle_diff = min(angle_diff, 360 - angle_diff)
                curvature_scores.append(angle_diff)

            # 点级别自适应简化
            for i in range(len(main_contour)):
                if i == 0 or i == len(main_contour) - 1:
                    # 保留起点和终点
                    contour_points.append(main_contour[i][0].tolist())
                    continue

                # 获取局部曲率
                prev_score = curvature_scores[i - 1] if i >= 1 else 0
                curr_score = curvature_scores[i] if i < len(curvature_scores) else 0
                max_curvature = max(prev_score, curr_score)

                # 在平直区域应用更强的简化
                epsilon = base_epsilon
                if max_curvature < 60:  # 小角度
                    # 创建局部片段
                    start_idx = max(0, i - 1)
                    end_idx = min(len(main_contour), i + 2)
                    segment = main_contour[start_idx:end_idx]

                    # 简化局部片段
                    if len(segment) > 2:
                        segment = segment.reshape(-1, 1, 2)
                        simplified = cv2.approxPolyDP(segment, epsilon, False)

                        # 只保留中间点
                        if len(simplified) == 3:
                            contour_points.append(simplified[1][0].tolist())
                    else:
                        contour_points.append(main_contour[i][0].tolist())
                else:
                    # 在拐点处保留原始点
                    contour_points.append(main_contour[i][0].tolist())
        else:
            # 简化短轮廓
            contour_points = main_contour.squeeze().tolist()

        # 确保格式正确
        if len(contour_points) > 0 and not isinstance(contour_points[0], list):
            contour_points = [contour_points]

        # 轮廓点后处理 - 移除异常点
        if len(contour_points) > 2:
            filtered_points = []
            max_distance = min(img_size) * 0.05  # 最大允许距离（图像尺寸的5%）

            for i in range(len(contour_points)):
                current_point = contour_points[i]

                if len(filtered_points) > 0:
                    prev_point = filtered_points[-1]
                    distance = np.sqrt((current_point[0] - prev_point[0]) ** 2 +
                                       (current_point[1] - prev_point[1]) ** 2)

                    if distance < max_distance:
                        filtered_points.append(current_point)
                    else:
                        # 添加中点作为过渡点
                        mid_point = [
                            (prev_point[0] + current_point[0]) / 2,
                            (prev_point[1] + current_point[1]) / 2
                        ]
                        filtered_points.append(mid_point)
                        filtered_points.append(current_point)
                else:
                    filtered_points.append(current_point)

            contour_points = filtered_points

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

        if len(contour_points) > 0:
            start_point = tuple(map(int, contour_points[0]))
            cv2.circle(overlay, start_point, 8, (255, 0, 0), -1)
            cv2.putText(overlay, "Start", (start_point[0] + 10, start_point[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        cv2.putText(overlay, f"Conf: {boxes[0]['confidence']:.2f}",
                    (bbox_x1, bbox_y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # 添加点数量显示
        cv2.putText(overlay, f"Points: {len(contour_points)}",
                    (10, image.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return contour_data, overlay

    def draw_fixed_contour(image, fixed_contour, original_size):
        """在图像上绘制固定轮廓"""
        overlay = image.copy()

        if not fixed_contour:
            return overlay

        # 转换为可绘制的格式
        points_array = np.array(fixed_contour, dtype=np.int32).reshape((-1, 1, 2))

        # 绘制轮廓
        cv2.drawContours(overlay, [points_array], -1, (0, 255, 255), 3)

        # 标记轮廓起点
        if len(points_array) > 0:
            start_point = tuple(points_array[0][0])
            cv2.circle(overlay, start_point, 8, (255, 100, 0), -1)
            cv2.putText(overlay, "Fixed", (start_point[0] + 10, start_point[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 0), 2)

        # 添加固定轮廓标记
        cv2.putText(overlay, "Fixed Contour", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        return overlay

    def save_contour_data(self,contour_data, frame_count, is_fixed=False):
        """保存轮廓数据到JSON文件"""
        if contour_data is None:
            return

        prefix = "fixed_" if is_fixed else ""
        filename = os.path.join(self.config.OUTPUT_DIR, f"{prefix}contour_{frame_count}.json")
        with open(filename, 'w') as f:
            json.dump(contour_data, f, indent=2)

        print(f"💾 轮廓数据已保存到: {filename}")

    def draw_masks_on_image(self, image, boxes, mask_coeffs, mask_protos, img_size):
        """掩码叠加"""
        masks = []
        for coeff in mask_coeffs:
            mask = np.tensordot(coeff.astype(np.float32), mask_protos.astype(np.float32), axes=([0], [0]))
            mask = 1 / (1 + np.exp(-mask))
            mask = cv2.resize(mask, img_size)
            mask = (mask > 0.5).astype(np.uint8) * 255
            masks.append(mask)

        for m in masks:
            color = np.random.randint(0, 255, (3,), dtype=np.uint8)
            overlay = np.zeros_like(image)
            for c in range(3):
                overlay[:, :, c] = m * color[c] // 255
            image = cv2.addWeighted(image, 1.0, overlay, 0.5, 0)
        return image

    def init_aruco(self):
        """初始化ArUco检测器"""
        # try:
        #     # 使用新版本的ArUco API
        #     self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        #     self.aruco_params = cv2.aruco.DetectorParameters()
        # except AttributeError:
        #     # 对于旧版本的OpenCV使用旧API
        #     self.aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
        #     self.aruco_params = cv2.aruco.DetectorParameters_create()

        # 对于旧版本的OpenCV使用旧API
        self.aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
        self.aruco_params = cv2.aruco.DetectorParameters_create()

    def load_model(self,model_path):
        """加载ONNX模型"""
        if not model_path or not os.path.exists(model_path):
            raise ValueError(f"无效的模型路径: {model_path}")
        try:
            self.model_path = model_path
            print(f"[INFO] 加载模型: {self.model_path}")
            self.session = ort.InferenceSession(
                self.model_path,
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )
            print("Providers:", self.session.get_providers())
            self.input_name = self.session.get_inputs()[0].name
            self.output_names = [o.name for o in self.session.get_outputs()]
            print(f"[INFO] 模型加载成功! 输入名: {self.input_name}, 输出名: {self.output_names}")

            # 获取模型输入尺寸
            input_info = self.session.get_inputs()[0]
            input_shape = input_info.shape
            if len(input_shape) == 4:  # [batch, channel, height, width]
                height = input_shape[2]
                width = input_shape[3]
                self.input_size = (width, height)
                print(f"[INFO] 模型输入尺寸为: {self.input_size}")
            elif len(input_shape) == 3:  # [batch, height, width]
                height = input_shape[1]
                width = input_shape[2]
                self.input_size = (width, height)
                print(f"[INFO] 模型输入尺寸为: {self.input_size}")
            else:
                print(f"[WARNING] 无法识别的输入形状: {input_shape}, 使用默认尺寸")
                self.input_size = (640, 640)

            # 动态检测类别数量（关键更新）
            output_info = self.session.get_outputs()[0]
            output_shape = output_info.shape
            if len(output_shape) >= 2:
                # 输出形状为 [batch, 84, 8400]
                # 84 = (x, y, w, h, obj_conf) + num_classes + num_mask
                self.num_classes = output_shape[1] - 5 - self.num_mask
                print(f"[INFO] 检测到 {self.num_classes} 个类别")
            else:
                self.num_classes = 80
                print("[WARNING] 无法确定类别数量，使用默认值80")

            return True
        except Exception as e:
            print(f"[ERROR] 加载模型失败: {str(e)}")
            return False

    def start_receiving(self):
        """启动接收线程"""
        if self.receive_thread and self.receive_thread.is_alive():
            return True

        self.receive_thread = threading.Thread(target=self.receive_video_thread, daemon=True)
        self.receive_thread.start()
        print("[SERVER] 接收线程已启动")
        return True

    def save_calibration_params(self, file_path=None):
        """保存标定参数到文件"""
        #file_path = file_path or os.path.join(self.config.CALIBRATION_FILE, file_path)
        try:
            file_path = self.config.CALIBRATION_FILE
            # os.path.join 拼接路径
            params = {
                'c_x': self.c_x,
                'c_y': self.c_y,
                'ratio': self.ratio,
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
            }
            with open(file_path, 'w') as f:
                json.dump(params, f, indent=4)
            print(f"[CALIBRATION] 标定参数已保存到: {file_path}")
            return True
        except Exception as e:
            print(f"[ERROR] 保存标定参数失败: {str(e)}")
            return False

    def load_calibration_params(self, file_path=None):
        """从文件加载标定参数"""
        try:
            file_path = file_path or os.path.join(self.config.CALIBRATION_FILE, file_path)
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    params = json.load(f)
                self.c_x = params.get('c_x', 0)
                self.c_y = params.get('c_y', 0)
                self.ratio = params.get('ratio', 0)
                self.calibration_complete = True
                print(f"[CALIBRATION] 标定参数已加载: c_x={self.c_x}, c_y={self.c_y}, ratio={self.ratio}")
                return True
            else:
                print("[CALIBRATION] 未找到标定文件")
                return False
        except Exception as e:
            print(f"[ERROR] 加载标定参数失败: {str(e)}")
            return False

    def receive_video_thread(self):
        """在子线程中接收视频帧"""
        try:
            server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server_socket.bind((self.server_ip, self.server_port))
            server_socket.listen(5)
            print(f"[SERVER] 服务端已启动，正在监听 {self.server_ip}:{self.server_port}")

            client_socket, client_addr = server_socket.accept()
            print(f"[SERVER] 已接受来自 {client_addr} 的连接")
            payload_size = struct.calcsize('>I')

            data = b""
            while not self.stop_event.is_set():
                try:
                    # 读取帧长度信息
                    while len(data) < payload_size:
                        packet = client_socket.recv(4096)
                        if not packet:
                            print("[SERVER] 客户端断开连接")
                            break
                        data += packet

                    if len(data) < payload_size:
                        print("[SERVER] 客户端断开连接")
                        break

                    packed_msg_size = data[:payload_size]
                    data = data[payload_size:]
                    msg_size = struct.unpack('>I', packed_msg_size)[0]

                    # 读取完整的帧数据
                    while len(data) < msg_size:
                        to_receive = min(4096, msg_size - len(data))
                        packet = client_socket.recv(to_receive)
                        if not packet:
                            print("[SERVER] 客户端断开连接")
                            break
                        data += packet

                    if len(data) < msg_size:
                        print("[SERVER] 连接中断，未能接收完整帧")
                        break

                    # 分割出当前帧数据
                    frame_data = data[:msg_size]
                    data = data[msg_size:]
                    frame_array = np.frombuffer(frame_data, dtype=np.uint8)
                    frame = cv2.imdecode(frame_array, flags=cv2.IMREAD_COLOR)
                    if frame is not None:
                        self.image_height, self.image_width = frame.shape[:2]
                        self.frame_queue.put(frame.copy(), block=True, timeout=0.5)
                    if frame is None:
                        print("[SERVER] 解码帧失败")
                        continue

                    # 将帧放入队列供主线程使用
                    try:
                        self.frame_queue.put(frame.copy(), block=True, timeout=0.5)
                    except queue.Full:
                        # 如果队列满了，跳过此帧
                        print("[SERVER] 帧队列已满，跳过帧")
                except socket.timeout:
                    print("[SERVER] 套接字超时")
                    continue
                except socket.error as e:
                    if not self.stop_event.is_set():
                        print(f"[SERVER] 套接字错误: {str(e)}")
                    break
                except Exception as e:
                    print(f"[SERVER] 接收错误: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    break

        except Exception as e:
            print(f"[SERVER] 接收线程发生错误: {str(e)}")
            import traceback
            traceback.print_exc()
        finally:
            print("[SERVER] 清理接收线程资源")
            if 'client_socket' in locals():
                client_socket.close()
            server_socket.close()
            print("[SERVER] 接收线程已停止")

    def detect_aruco_markers(self, img):
        """检测ArUco标记并绘制到图像上，返回检测结果"""
        if not self.calibration_complete or self.calibration_in_progress:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            try:
                # 尝试使用新版本的ArUco API
                detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
                corners, ids, _ = detector.detectMarkers(gray)
            except (AttributeError, TypeError):
                # 对于旧版本的OpenCV使用旧API
                corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)

            if ids is None or len(ids) < 2:
                return None

            marker_positions = []
            for i, corner in enumerate(corners):
                pts = corner[0]
                x = int(np.mean(pts[:, 0]))
                y = int(np.mean(pts[:, 1]))
                marker_positions.append((ids[i][0], x, y))

            # 按ID排序以确保一致性
            marker_positions.sort(key=lambda x: x[0])

            if len(marker_positions) < 2:
                return None

            _, x1, y1 = marker_positions[0]
            _, x2, y2 = marker_positions[1]

            # 绘制ArUco标记
            try:
                # 新版本绘制方法
                cv2.aruco.drawDetectedMarkers(img, corners, ids)
            except (AttributeError, TypeError):
                # 旧版本绘制方法
                cv2.aruco.drawDetectedMarkers(img, corners, ids)

            # 在标记中心绘制圆点
            cv2.circle(img, (x1, y1), 10, (0, 255, 0), 2)
            cv2.circle(img, (x2, y2), 10, (0, 255, 0), 2)
            cv2.putText(img, f"Marker 1", (x1 - 40, y1 - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(img, f"Marker 2", (x2 - 40, y2 - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            return x1, x2, y1, y2

        return None

    def pixel_to_world_coords(self, pixel_x, pixel_y, rotation=0):
        """
        将像素坐标转换为世界坐标系坐标
        """
        if not self.calibration_complete:
            return 0, 0, False

        if not hasattr(self, 'image_width') or not hasattr(self, 'image_height'):
            self.image_width = 640
            self.image_height = 480

        # 处理坐标系旋转
        if rotation == 90:
            # 顺时针旋转90度
            temp = pixel_x
            pixel_x = self.image_height - pixel_y
            pixel_y = temp
        elif rotation == 180:
            pixel_x = self.image_width - pixel_x
            pixel_y = self.image_height - pixel_y
        elif rotation == 270:
            # 逆时针旋转90度
            temp = pixel_x
            pixel_x = pixel_y
            pixel_y = self.image_width - temp

        # 应用标定参数
        offset_x = (pixel_y - self.c_y) * self.ratio
        offset_y = (pixel_x - self.c_x) * self.ratio

        world_x = offset_x + self.camera_offset_x
        world_y = offset_y + self.camera_offset_y

        return world_x, world_y, True

    def non_max_suppression(self, boxes, scores, iou_threshold):
        """执行非极大值抑制，过滤重叠的检测框"""
        if len(boxes) == 0:
            return []

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)

        idxs = scores.argsort()[::-1]

        keep = []

        while len(idxs) > 0:
            i = idxs[0]
            keep.append(i)

            xx1 = np.maximum(x1[i], x1[idxs[1:]])
            yy1 = np.maximum(y1[i], y1[idxs[1:]])
            xx2 = np.minimum(x2[i], x2[idxs[1:]])
            yy2 = np.minimum(y2[i], y2[idxs[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)

            intersection = w * h
            iou = intersection / (areas[i] + areas[idxs[1:]] - intersection)

            mask = iou <= iou_threshold
            idxs = idxs[1:][mask]

        return keep

    def draw_fixed_contour(self, image, fixed_contour, original_size):
        """在图像上绘制固定轮廓（新增方法）"""
        overlay = image.copy()
        w, h = original_size

        if not fixed_contour:
            return overlay

        # 转换为可绘制的格式
        points_array = np.array(fixed_contour, dtype=np.int32).reshape((-1, 1, 2))

        # 绘制轮廓
        cv2.drawContours(overlay, [points_array], -1, (0, 255, 255), 3)  # 青色轮廓表示固定轮廓

        # 标记轮廓起点
        if len(points_array) > 0:
            start_point = tuple(points_array[0][0])
            cv2.circle(overlay, start_point, 8, (255, 100, 0), -1)  # 橙蓝色点
            cv2.putText(overlay, "Fixed", (start_point[0] + 10, start_point[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 0), 2)

        # 添加固定轮廓标记
        cv2.putText(overlay, "Fixed Contour", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        return overlay

    def save_contour_data(self, contour_data, frame_count, is_fixed=False):
        """保存轮廓数据到JSON文件（新增方法）"""
        if contour_data is None:
            return

        # 确保输出目录存在
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        prefix = "fixed_" if is_fixed else ""
        filename = os.path.join(self.output_dir, f"{prefix}contour_{frame_count}.json")
        with open(filename, 'w') as f:
            json.dump(contour_data, f, indent=2)

        print(f"💾 轮廓数据已保存到: {filename}")

    def detect_objects(self, frame):
        """检测图像中的对象（整合轮廓处理和固定轮廓）"""
        if not self.session or not self.calibration_complete:
            return frame, [], []

        # 帧计数器递增
        self.frame_counter += 1

        try:
            # 使用更新后的推理函数
            boxes, mask_coeffs, mask_protos, img_size = self.infer(frame)
            orig_w, orig_h = img_size
            detections = []
            world_coords = []

            # 处理轮廓
            contour_data = None
            if boxes:
                box = boxes[0]
                mask = mask_coeffs[0] if mask_coeffs else None

                # 计算边界框坐标
                x, y, w, h = box['bbox']
                x1 = int(max(0, min(x - w / 2, orig_w - 1)))
                y1 = int(max(0, min(y - h / 2, orig_h - 1)))
                x2 = int(max(0, min(x + w / 2, orig_w - 1)))
                y2 = int(max(0, min(y + h / 2, orig_h - 1)))

                # 计算中心点
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2

                # 转换到世界坐标
                world_x, world_y, success = self.pixel_to_world_coords(center_x, center_y)

                # 处理掩膜和轮廓（新增轮廓处理）
                if mask is not None and mask_protos is not None:
                    contour_data, frame = self.process_contour(
                        frame, box, [mask], mask_protos, (orig_w, orig_h)
                    )

                    # 从轮廓数据中获取更精确的中心点
                    if contour_data and contour_data.get("contour_points"):
                        contour_points = np.array(contour_data["contour_points"])
                        contour_center = np.mean(contour_points, axis=0).astype(int)
                        center_x, center_y = contour_center

                        # 使用轮廓中心重新计算世界坐标
                        world_x, world_y, success = self.pixel_to_world_coords(center_x, center_y)

                        # 保存当前轮廓作为固定轮廓的选项
                        if self.save_as_fixed:
                            self.fixed_contour = contour_points.tolist()
                            self.save_as_fixed = False  # 重置标志
                else:
                    # 如果没有掩膜数据，使用边界框绘制
                    color = (0, 255, 0)  # 绿色边框
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    label_text = f"{box['class_id']}:{box['confidence']:.2f}"
                    cv2.putText(frame, label_text, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # 绘制中心点和坐标
                cv2.circle(frame, (center_x, center_y), 5, (0, 255, 255), -1)
                if success:
                    coord_text = f"({world_x:.1f}, {world_y:.1f}) mm"
                    cv2.putText(frame, coord_text, (center_x + 10, center_y - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                # 记录检测信息
                detections.append({
                    'center': (center_x, center_y),
                    'world_coords': (world_x, world_y),
                    'bbox': (x1, y1, x2, y2),
                    'label': box['class_id'],
                    'score': box['confidence'],
                    'contour': contour_data  # 轮廓数据
                })
                world_coords.append((world_x, world_y))

                # 保存轮廓数据
                if contour_data and self.save_contour_data:
                    self.save_contour_data(contour_data, self.frame_counter)

            # 绘制固定轮廓
            if self.fixed_contour:
                frame = self.draw_fixed_contour(frame, self.fixed_contour, (orig_w, orig_h))

                # 保存固定轮廓数据
                if self.save_fixed_contour:
                    self.save_contour_data({
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "fixed_contour": True,
                        "contour_points": self.fixed_contour,
                        "image_size": (orig_w, orig_h)
                    }, self.frame_counter, is_fixed=True)
                    self.save_fixed_contour = False  # 重置标志

            return frame, detections, world_coords

        except Exception as e:
            print(f"目标检测错误: {str(e)}")
            import traceback
            traceback.print_exc()
            return frame, [], []

    def process_contour(self, image, box, masks, protos, img_size):
        """处理轮廓数据（基于新函数实现）"""
        # 获取目标边界框信息
        x, y, w, h = box['bbox']
        bbox_x1 = int(x - w / 2)
        bbox_y1 = int(y - h / 2)
        bbox_x2 = int(x + w / 2)
        bbox_y2 = int(y + h / 2)
        bbox_area = w * h

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
        kernel = np.ones((3, 3), np.uint8)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)

        # 查找所有轮廓
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, image

        # 筛选有效轮廓
        valid_contours = []
        for contour in contours:
            contour_x, contour_y, contour_w, contour_h = cv2.boundingRect(contour)
            contour_x2 = contour_x + contour_w
            contour_y2 = contour_y + contour_h

            overlap_x1 = max(bbox_x1, contour_x)
            overlap_y1 = max(bbox_y1, contour_y)
            overlap_x2 = min(bbox_x2, contour_x2)
            overlap_y2 = min(bbox_y2, contour_y2)

            overlap_width = max(0, overlap_x2 - overlap_x1)
            overlap_height = max(0, overlap_y2 - overlap_y1)
            overlap_area = overlap_width * overlap_height

            contour_area = cv2.contourArea(contour)

            if overlap_area > 0.5 * bbox_area and 0.1 * bbox_area < contour_area < 2.0 * bbox_area:
                valid_contours.append(contour)

        if not valid_contours:
            return None, image

        # 处理主轮廓
        main_contour = max(valid_contours, key=cv2.contourArea)
        contour_points = []

        # 自适应轮廓简化
        if main_contour is not None and len(main_contour) > 4:
            contour_length = cv2.arcLength(main_contour, True)
            base_epsilon = 0.005 * contour_length

            # 计算曲率变化
            curvature_scores = []
            for i in range(1, len(main_contour) - 1):
                p1 = main_contour[i - 1][0]
                p2 = main_contour[i][0]
                p3 = main_contour[i + 1][0]

                vec1 = (p1[0] - p2[0], p1[1] - p2[1])
                vec2 = (p3[0] - p2[0], p3[1] - p2[1])

                angle1 = np.arctan2(vec1[1], vec1[0])
                angle2 = np.arctan2(vec2[1], vec2[0])
                angle_diff = np.abs(np.degrees(angle1 - angle2))
                angle_diff = min(angle_diff, 360 - angle_diff)
                curvature_scores.append(angle_diff)

            # 点级别自适应简化
            for i in range(len(main_contour)):
                if i == 0 or i == len(main_contour) - 1:
                    contour_points.append(main_contour[i][0].tolist())
                    continue

                # 获取局部曲率
                prev_score = curvature_scores[i - 1] if i >= 1 else 0
                curr_score = curvature_scores[i] if i < len(curvature_scores) else 0
                max_curvature = max(prev_score, curr_score)

                epsilon = base_epsilon
                if max_curvature < 60:
                    start_idx = max(0, i - 1)
                    end_idx = min(len(main_contour), i + 2)
                    segment = main_contour[start_idx:end_idx]

                    if len(segment) > 2:
                        segment = segment.reshape(-1, 1, 2)
                        simplified = cv2.approxPolyDP(segment, epsilon, False)

                        if len(simplified) == 3:
                            contour_points.append(simplified[1][0].tolist())
                    else:
                        contour_points.append(main_contour[i][0].tolist())
                else:
                    contour_points.append(main_contour[i][0].tolist())
        else:
            contour_points = main_contour.squeeze().tolist()

        # 确保格式正确
        if len(contour_points) > 0 and not isinstance(contour_points[0], list):
            contour_points = [contour_points]

        # === 新增: 固定间距采样 ===
        if len(contour_points) > 1:
            fixed_distance_contour = []
            # 计算固定距离点 (5像素)
            fixed_distance = 5

            # 添加起点
            start_point = np.array(contour_points[0])
            fixed_distance_contour.append(start_point.tolist())
            segment_start = start_point
            remaining_distance = fixed_distance

            # 遍历所有轮廓点
            for i in range(1, len(contour_points)):
                current_point = np.array(contour_points[i])
                segment_vector = current_point - segment_start
                segment_length = np.linalg.norm(segment_vector)

                # 如果当前段长度大于剩余距离
                while segment_length > remaining_distance:
                    # 计算新点位置
                    ratio = remaining_distance / segment_length
                    new_point = segment_start + ratio * segment_vector
                    fixed_distance_contour.append(new_point.tolist())

                    # 更新剩余距离和起点
                    segment_start = new_point
                    segment_vector = current_point - segment_start
                    segment_length = np.linalg.norm(segment_vector)
                    remaining_distance = fixed_distance
                else:
                    # 更新剩余距离
                    remaining_distance -= segment_length
                    segment_start = current_point

            # 确保闭合（如果起点和终点不同）
            if len(fixed_distance_contour) > 2:
                last_point = np.array(fixed_distance_contour[-1])
                first_point = np.array(fixed_distance_contour[0])
                distance_to_start = np.linalg.norm(last_point - first_point)

                if distance_to_start > fixed_distance * 0.5:  # 如果不接近起点
                    segment_vector = first_point - last_point
                    segment_length = np.linalg.norm(segment_vector)

                    # 添加点到起点
                    while segment_length > remaining_distance:
                        ratio = remaining_distance / segment_length
                        new_point = last_point + ratio * segment_vector
                        fixed_distance_contour.append(new_point.tolist())
                        last_point = new_point
                        segment_vector = first_point - last_point
                        segment_length = np.linalg.norm(segment_vector)
                        remaining_distance = fixed_distance

                    # 添加起点
                    fixed_distance_contour.append(first_point.tolist())

            # 使用固定间距的点替换原始点
            contour_points = fixed_distance_contour
            print(f"固定间距轮廓点生成 | 点数: {len(contour_points)} | 间距: {fixed_distance}像素")

        # 创建轮廓数据结构
        contour_data = {
            "confidence": box['confidence'],
            "class_id": box['class_id'],
            "bbox": box['bbox'],
            "contour_points": contour_points,
            "image_size": img_size
        }

        # 准备轮廓点用于绘制
        contour_array = np.array(contour_points, dtype=np.int32).reshape((-1, 1, 2))

        # 绘制轮廓
        overlay = image.copy()
        cv2.rectangle(overlay, (bbox_x1, bbox_y1), (bbox_x2, bbox_y2), (0, 255, 0), 2)
        cv2.drawContours(overlay, [contour_array], -1, (0, 0, 255), 3)

        if len(contour_points) > 0:
            start_point = tuple(map(int, contour_points[0]))
            cv2.circle(overlay, start_point, 8, (255, 0, 0), -1)
            cv2.putText(overlay, "Start", (start_point[0] + 10, start_point[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        cv2.putText(overlay, f"Conf: {box['confidence']:.2f}",
                    (bbox_x1, bbox_y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # 添加点数量显示
        cv2.putText(overlay, f"Points: {len(contour_points)}",
                    (10, image.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return contour_data, overlay

    def stop(self):
        """停止所有操作"""
        self.stop_event.set()
        if self.receive_thread and self.receive_thread.is_alive():
            self.receive_thread.join(timeout=2.0)
        cv2.destroyAllWindows()
        print("[SYSTEM] 摄像头检测系统已停止")