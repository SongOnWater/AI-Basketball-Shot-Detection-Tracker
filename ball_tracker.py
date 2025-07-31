"""
Ball tracking module for basketball shot detection
"""
import numpy as np
import math
from utils import clean_ball_pos


class BallTracker:
    """
    Handles ball tracking and trajectory prediction
    """
    def __init__(self, min_ball_area=100, debug_logger=None):
        self.min_ball_area = min_ball_area
        self.ball_pos = []
        self.debug_logger = debug_logger
        self.selected_ball = None
        
    def is_reasonable_ball_position(self, ball_data, frame_height=None):
        """
        Check if ball position is reasonable to filter out false positives

        Args:
            ball_data: Ball detection dictionary with 'center' key
            frame_height: Maximum reasonable frame height. If None, will use current frame height.

        Returns:
            bool: True if position is reasonable, False otherwise
        """
        if frame_height is None:
            # We don't have access to frame here, so we'll skip this check in this refactored version
            # The check will be done in the main detector class
            return True

        center_y = ball_data['center'][1]

        # Position reasonableness check
        if center_y > frame_height * 1.1:  # Allow 10% margin above typical height
            if self.debug_logger:
                self.debug_logger.warning(f"Ball Y position {center_y} exceeds reasonable limit {frame_height * 1.1}")
            return False

        return True
        
    def predict_ball_position_from_trajectory(self, current_frame):
        """
        预测当前帧中球的位置，基于历史轨迹数据拟合
        
        Args:
            current_frame: 当前帧号
            
        Returns:
            tuple: 预测的球位置 (x, y) 或 None（如果历史数据不足）
        """
        # 需要至少3个历史点来进行有效的轨迹拟合
        if len(self.ball_pos) < 3:
            if self.debug_logger:
                self.debug_logger.debug_file_only(f"历史轨迹点不足，无法进行拟合预测: {len(self.ball_pos)} 点")
            return None
            
        # 获取最近的N个历史点（最多10个点）
        recent_history = self.ball_pos[-10:]
        
        # 提取坐标和帧号
        x_coords = [pos[0][0] for pos in recent_history]
        y_coords = [pos[0][1] for pos in recent_history]
        frames = [pos[1] for pos in recent_history]
        
        # 检查是否有足够的不同帧
        if len(set(frames)) < 3:
            if self.debug_logger:
                self.debug_logger.debug_file_only(f"历史轨迹中不同帧数不足，无法进行拟合预测")
            return None
            
        try:
            # 对X坐标进行多项式拟合（2阶）
            x_poly = np.polyfit(frames, x_coords, 2)
            x_poly_func = np.poly1d(x_poly)
            
            # 对Y坐标进行多项式拟合（2阶）
            y_poly = np.polyfit(frames, y_coords, 2)
            y_poly_func = np.poly1d(y_poly)
            
            # 预测当前帧的位置
            predicted_x = x_poly_func(current_frame)
            predicted_y = y_poly_func(current_frame)
            
            if self.debug_logger:
                self.debug_logger.debug_file_only(f"轨迹拟合预测位置: ({predicted_x:.1f}, {predicted_y:.1f}) 在帧 {current_frame}")
            
            return (predicted_x, predicted_y)
        except Exception as e:
            if self.debug_logger:
                self.debug_logger.warning(f"轨迹拟合预测失败: {e}")
            return None

    def clean_motion(self, frame_count):
        """
        Clean up motion data by removing inaccurate data points
        """
        self.ball_pos = clean_ball_pos(self.ball_pos, frame_count)
        
    def add_ball_position(self, ball_data, frame_count):
        """
        Add a new ball position to the tracking history
        
        Args:
            ball_data: Ball detection data
            frame_count: Current frame number
        """
        self.ball_pos.append((
            (ball_data['center'][0], ball_data['center'][1]),
            frame_count,
            ball_data['size']['width'],
            ball_data['size']['height'],
            ball_data['confidence']
        ))
        
    def reset_tracking(self):
        """
        Reset ball tracking data
        """
        self.ball_pos = []
        self.selected_ball = None