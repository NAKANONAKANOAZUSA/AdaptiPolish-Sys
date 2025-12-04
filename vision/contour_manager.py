import numpy as np
from collections import deque
from scipy.signal import savgol_filter
from scipy.interpolate import CubicSpline
import scipy.ndimage as ndi
HISTORY_FRAMES = 5  # 收集前5次识别到的数据
class FixedContourManager:
    """固定轮廓管理器"""
    def __init__(self, config,max_frames=HISTORY_FRAMES):
        """初始化固定轮廓管理器"""
        self.config=config
        self.max_frames = max_frames
        self.raw_contours = deque(maxlen=max_frames)  # 使用deque存储原始轮廓数据
        self.smoothed_contour = None  # 平滑后的轮廓
        self.smoothed_size = None  # 平滑轮廓对应的图像尺寸
        self.completed = False  # 是否完成固定轮廓生成
        self.tracking_id = None  # 跟踪的目标ID
    
    def add_contour(self, contour_data):
        """添加新的轮廓数据到管理器"""
        # 如果已完成或数据无效，直接返回
        if self.completed or contour_data is None or not contour_data.get('contour_points'):
            return
            
        # 首次添加时记录目标ID
        if not self.raw_contours:
            self.tracking_id = contour_data.get('class_id', -1)
            print(f"开始跟踪目标ID: {self.tracking_id}")
        
        # 仅添加相同ID的轮廓
        if contour_data.get('class_id', -1) == self.tracking_id:
            self.raw_contours.append(contour_data)
            print(f"添加轮廓数据 | 当前数量: {len(self.raw_contours)}/{self.max_frames}")
            
            # 检查是否达到最大帧数
            if len(self.raw_contours) >= self.max_frames:
                print("收集完成，开始计算固定轮廓")
                self.completed = True
                self._calculate_fixed_contour()
    
    def get_fixed_contour(self):
        """获取最终固定轮廓"""
        return self.smoothed_contour, self.smoothed_size
    
    def _calculate_fixed_contour(self):
        """计算最终的固定轮廓"""
        if not self.raw_contours:
            print("无轮廓数据可用于计算固定轮廓")
            return
        
        print(f"开始计算固定轮廓 | 收集帧数: {len(self.raw_contours)}")
        
        # 获取最新轮廓的点数作为基准
        base_points = len(self.raw_contours[-1]['contour_points'])
        
        # 如果点数太少，则使用最后一个轮廓
        if base_points < self.config.MIN_CONTOUR_POINTS:
            print("点数太少，直接使用最后一个轮廓")
            self.smoothed_contour = self.raw_contours[-1]['contour_points']
            self.smoothed_size = self.raw_contours[-1]['image_size']
            self.completed = True
            return
        
        # 调整所有轮廓点数为相同长度
        aligned_contours = []
        for contour in self.raw_contours:
            current_points = np.array(contour['contour_points'])
            aligned = self._align_point_count(current_points, self.config.TARGET_POINTS)
            aligned_contours.append(aligned)
        
        # 计算每个点的中间值
        stacked = np.stack(aligned_contours)
        fixed_contour = np.median(stacked, axis=0)
        
        # 应用高级平滑处理
        smoothed_contour = self._apply_advanced_smoothing(fixed_contour)
            
        self.smoothed_contour = smoothed_contour.tolist()
        self.smoothed_size = self.raw_contours[-1]['image_size']
        self.completed = True
        
        print(f"固定轮廓生成完成 | 点数: {len(self.smoothed_contour)}")
    
    def _align_point_count(self, points, target_count):
        """重采样轮廓点使其具有相同点数"""
        if len(points) == target_count:
            return points
        
        # 计算轮廓周长
        contour_length = np.sum(np.linalg.norm(np.diff(points, axis=0, append=points[0:1]), axis=1))
        
        # 生成线性参数化
        t_current = np.linspace(0, contour_length, len(points))
        t_target = np.linspace(0, contour_length, target_count)
        
        # 使用三次样条插值保持曲线平滑
        if len(points) > 3:
            cs_x = CubicSpline(t_current, points[:, 0])
            cs_y = CubicSpline(t_current, points[:, 1])
            x_interp = cs_x(t_target)
            y_interp = cs_y(t_target)
        else:
            # 点数太少时使用线性插值
            x_interp = np.interp(t_target, t_current, points[:, 0])
            y_interp = np.interp(t_target, t_current, points[:, 1])
        
        return np.column_stack((x_interp, y_interp))
    
    def _apply_advanced_smoothing(self, contour):
        """应用高级平滑算法"""
        # Savitzky-Golay滤波器
        if len(contour) > 7:
            window_size = min(7, len(contour) // 2 * 2 + 1)  # 确保为奇数
            x_smooth = savgol_filter(contour[:, 0], window_size, 3)
            y_smooth = savgol_filter(contour[:, 1], window_size, 3)
            contour = np.column_stack((x_smooth, y_smooth))
        
        # 高斯平滑
        x_smooth = ndi.gaussian_filter1d(contour[:, 0], sigma=self.config.SMOOTH_SIGMA)
        y_smooth = ndi.gaussian_filter1d(contour[:, 1], sigma=self.config.SMOOTH_SIGMA)
        
        # 边界平滑
        if len(contour) > 10:
            # 创建周期性轮廓以平滑边界
            periodic_contour = np.vstack((contour, contour[:5]))
            x_periodic = ndi.gaussian_filter1d(periodic_contour[:, 0], sigma=self.config.SMOOTH_SIGMA)
            y_periodic = ndi.gaussian_filter1d(periodic_contour[:, 1], sigma=self.config.SMOOTH_SIGMA)
            
            # 取中间部分
            smoothed_contour = np.column_stack((
                x_periodic[2:-2],
                y_periodic[2:-2]
            ))
        else:
            smoothed_contour = np.column_stack((x_smooth, y_smooth))
        
        return smoothed_contour
    
    def reset(self):
        """重置管理器状态"""
        self.raw_contours.clear()
        self.completed = False
        self.tracking_id = None
        self.smoothed_contour = None
        self.smoothed_size = None
        print(" 轮廓管理器已重置")