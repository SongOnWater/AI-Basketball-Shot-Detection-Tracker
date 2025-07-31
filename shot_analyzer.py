"""
Shot analysis module for basketball shot detection
"""
import math
from utils import score


class ShotAnalyzer:
    """
    Handles shot detection and analysis
    """
    def __init__(self, debug_logger=None):
        self.debug_logger = debug_logger
        self.up = False
        self.down = False
        self.up_frame = 0
        self.down_frame = 0
        self.makes = 0
        self.attempts = 0
        
        # Overlay properties
        self.overlay_color = (0, 0, 255)  # 默认红色
        self.overlay_text = "Waiting..."
        self.fade_counter = 0
        self.fade_frames = 30  # 默认淡出帧数
        
    def detect_up(self, selected_ball, selected_hoop):
        """
        Detect if ball is in UP region using synchronized data
        
        Args:
            selected_ball: Ball detection data from current frame
            selected_hoop: Hoop detection data from current frame
            
        Returns:
            bool: True if ball is in UP region
        """
        # Extract data
        ball_center = selected_ball[0]
        ball_x, ball_y = ball_center
        hoop_center = selected_hoop[0]
        hoop_x, hoop_y = hoop_center
        hoop_w = selected_hoop[2]
        hoop_h = selected_hoop[3]

        # Define UP region (around backboard/hoop area)
        x1 = hoop_x - 4 * hoop_w
        x2 = hoop_x + 4 * hoop_w
        y1 = hoop_y - 2 * hoop_h
        y2 = hoop_y - 0.5 * hoop_h

        # Check if ball is in UP region
        is_in_up_region = (x1 < ball_x < x2 and y1 < ball_y < y2)
        if self.debug_logger:
            self.debug_logger.debug(f"UP检测 - Frame: ball=({ball_x},{ball_y}), region=({x1:.1f},{y1:.1f}) to ({x2:.1f},{y2:.1f}), in_region={is_in_up_region}")
        
        return is_in_up_region

    def detect_down(self, selected_ball, selected_hoop):
        """
        Detect if ball is in DOWN region using synchronized data
        
        Args:
            selected_ball: Ball detection data from current frame
            selected_hoop: Hoop detection data from current frame
            
        Returns:
            bool: True if ball is in DOWN region
        """
        # Extract data
        ball_center = selected_ball[0]
        ball_x, ball_y = ball_center
        hoop_center = selected_hoop[0]
        hoop_x, hoop_y = hoop_center
        hoop_h = selected_hoop[3]

        # Define DOWN region (below hoop)
        rim_top_y = hoop_y - 0.5 * hoop_h
        
        # Check if ball is below the rim (in DOWN region)
        is_in_down_region = (ball_y > rim_top_y)
        if self.debug_logger:
            self.debug_logger.debug(f"DOWN检测 - Frame: ball_y={ball_y:.1f}, threshold={rim_top_y:.1f} (hoop top edge), in_region={is_in_down_region}")
        
        return is_in_down_region
        
    def analyze_shot_attempt(self, ball_pos, hoop_pos):
        """Analyze shot attempt with safe state handling"""
        if len(ball_pos) > 0 and len(hoop_pos) > 0:
            debug_info = {}

            # ✅ 使用局部值进行判断，防止状态被清零
            up_frame = self.up_frame
            down_frame = self.down_frame
            is_valid = self.up and self.down and up_frame < down_frame

            debug_info['shot_context'] = {
                'up_frame': up_frame,
                'down_frame': down_frame,
                'is_valid_sequence': is_valid
            }

            if is_valid:
                if self.debug_logger:
                    self.debug_logger.info(f"🔥 Valid shot sequence: UP@{up_frame} → DOWN@{down_frame}")
                self.attempts += 1
                is_successful = score(ball_pos, hoop_pos, debug_info)
                if is_successful:
                    self.makes += 1
                    self.overlay_color = (0, 255, 0)  # Green for make
                    self.overlay_text = "Make"
                    self.fade_counter = self.fade_frames
                else:
                    self.overlay_color = (255, 0, 0)  # Red for miss
                    self.overlay_text = "Miss"
                    self.fade_counter = self.fade_frames
            else:
                debug_info['failure_reason'] = "Invalid sequence or missing UP/DOWN states"
                is_successful = False
                self.overlay_color = (255, 0, 0)  # Red for miss
                self.overlay_text = "Miss"
                self.fade_counter = self.fade_frames

            # 🔧 状态只在**分析完成后**统一重置
            self.up = False
            self.down = False
            self.up_frame = 0
            self.down_frame = 0
            
            return {
                'is_successful': is_successful,
                'debug_info': debug_info
            }
            
        return {
            'is_successful': False,
            'debug_info': {'failure_reason': 'No ball or hoop positions'}
        }
        
    def reset_shot_detection(self):
        """
        Reset shot detection state
        """
        self.up = False
        self.down = False
        self.up_frame = 0
        self.down_frame = 0