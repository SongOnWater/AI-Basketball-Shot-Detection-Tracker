"""
Hoop tracking module for basketball shot detection
"""
import math
from utils import clean_hoop_pos


class HoopTracker:
    """
    Handles hoop tracking and related functionality
    """
    def __init__(self, debug_logger=None):
        self.hoop_pos = []
        self.selected_hoop = None
        self.debug_logger = debug_logger
        
    def clean_motion(self):
        """
        Clean up hoop position data
        """
        self.hoop_pos = clean_hoop_pos(self.hoop_pos)
        
    def add_hoop_position(self, hoop_data, frame_count):
        """
        Add a new hoop position to the tracking history
        
        Args:
            hoop_data: Hoop detection data
            frame_count: Current frame number
        """
        self.hoop_pos.append((
            (hoop_data['center'][0], hoop_data['center'][1]),
            frame_count,
            hoop_data['size']['width'],
            hoop_data['size']['height'],
            hoop_data['confidence']
        ))
        
    def reset_tracking(self):
        """
        Reset hoop tracking data
        """
        self.hoop_pos = []
        self.selected_hoop = None
        
    def is_ball_in_hoop_area(self, ball_pos, hoop_pos):
        """
        Check if ball position is within the hoop area
        
        Args:
            ball_pos: Ball position (x, y)
            hoop_pos: Hoop position data
            
        Returns:
            bool: True if ball is in hoop area
        """
        if not hoop_pos:
            return False
            
        hoop_center = hoop_pos[0]  # (x, y)
        hoop_width = hoop_pos[2]
        hoop_height = hoop_pos[3]
        
        # Define hoop area (a bit larger than the actual hoop to account for errors)
        x1 = hoop_center[0] - 0.6 * hoop_width
        x2 = hoop_center[0] + 0.6 * hoop_width
        y1 = hoop_center[1] - 0.6 * hoop_height
        y2 = hoop_center[1] + 0.6 * hoop_height
        
        ball_x, ball_y = ball_pos
        
        return x1 <= ball_x <= x2 and y1 <= ball_y <= y2