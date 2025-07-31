
import math
import numpy as np
import warnings
import torch

def get_device():
    """Automatically select devices -> mps（Mac） -> cpu"""
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    return device

def clean_ball_pos(ball_pos, frame_count):
    """
    Clean up ball position data by removing inaccurate data points
    
    Args:
        ball_pos: List of ball positions
        frame_count: Current frame count
        
    Returns:
        Cleaned ball position list
    """
    if len(ball_pos) <= 1:
        return ball_pos
        
    # Remove data points that are too old (older than 30 frames)
    while len(ball_pos) > 1 and frame_count - ball_pos[0][1] > 30:
        ball_pos.pop(0)
        
    return ball_pos

def clean_hoop_pos(hoop_pos):
    """
    Clean up hoop position data by removing inaccurate data points
    
    Args:
        hoop_pos: List of hoop positions
        
    Returns:
        Cleaned hoop position list
    """
    if len(hoop_pos) <= 1:
        return hoop_pos
        
    # Keep only the most recent hoop positions (maximum 10)
    if len(hoop_pos) > 10:
        hoop_pos = hoop_pos[-10:]
        
    return hoop_pos


def score(ball_pos, hoop_pos, debug_info=None):
    """
    Determine if a shot is successful and collect debug information
    
    Args:
        ball_pos: Ball position trajectory
        hoop_pos: Hoop position
        debug_info: Optional dictionary for storing debug information
    
    Returns:
        bool: Whether the shot was successful
    """
    # Initialize debug information dictionary
    if debug_info is None:
        debug_info = {}
    
    x = []
    y = []
    rim_height = hoop_pos[-1][0][1] - 0.5 * hoop_pos[-1][3]
    
    # Record hoop information
    debug_info['hoop_info'] = {
        'position': {'x': hoop_pos[-1][0][0], 'y': hoop_pos[-1][0][1]},
        'width': hoop_pos[-1][2],
        'height': hoop_pos[-1][3],
        'rim_height': rim_height
    }
    
    # Record ball trajectory points
    trajectory_points = []
    for i in range(len(ball_pos)):
        trajectory_points.append({
            'position': {'x': ball_pos[i][0][0], 'y': ball_pos[i][0][1]},
            'frame': ball_pos[i][1],
            'width': ball_pos[i][2],
            'height': ball_pos[i][3],
            'confidence': ball_pos[i][4] if len(ball_pos[i]) > 4 else None
        })
    debug_info['ball_trajectory'] = trajectory_points

    # Get first point above rim and first point below rim
    above_point = None
    below_point = None
    
    # Find last point above rim
    for i in reversed(range(len(ball_pos))):
        if ball_pos[i][0][1] < rim_height:
            x.append(ball_pos[i][0][0])
            y.append(ball_pos[i][0][1])
            above_point = {'x': ball_pos[i][0][0], 'y': ball_pos[i][0][1], 'frame': ball_pos[i][1]}
            break
    
    # Use the last point (DOWN frame detected point) as the first point below rim
    if len(ball_pos) > 0:
        last_ball_pos = ball_pos[-1]
        x.append(last_ball_pos[0][0])
        y.append(last_ball_pos[0][1])
        below_point = {'x': last_ball_pos[0][0], 'y': last_ball_pos[0][1], 'frame': last_ball_pos[1]}
    
    debug_info['key_points'] = {
        'above_rim_point': above_point,
        'below_rim_point': below_point
    }
    
    # If not enough points found to create trajectory line
    if len(x) <= 1:
        debug_info['failure_reason'] = "Not enough trajectory points to determine shot result"
        return False

    # Create line from two points
    try:
        # Suppress polyfit warnings as we're intentionally fitting with minimal points
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", np.RankWarning)
            m, b = np.polyfit(x, y, 1)
    except Exception:
        debug_info['failure_reason'] = "Failed to fit trajectory line"
        return False
    
    # Check if the slope is reasonable (avoid division by zero or near-zero)
    if abs(m) < 1e-10:
        debug_info['failure_reason'] = "Trajectory line too flat to determine shot result"
        return False
    
    predicted_x = ((hoop_pos[-1][0][1] - 0.5 * hoop_pos[-1][3]) - b) / m
    # Use smaller rim area for stricter judgment (0.3 instead of 0.4)
    rim_x1 = hoop_pos[-1][0][0] - 0.3 * hoop_pos[-1][2]
    rim_x2 = hoop_pos[-1][0][0] + 0.3 * hoop_pos[-1][2]
    
    # Record trajectory line and prediction information
    debug_info['trajectory_line'] = {
        'slope': float(m),
        'intercept': float(b),
        'equation': f"y = {m:.4f}x + {b:.4f}"
    }
    
    debug_info['prediction'] = {
        'predicted_x_at_rim': float(predicted_x),
        'rim_x1': float(rim_x1),
        'rim_x2': float(rim_x2),
        'rim_width': float(rim_x2 - rim_x1)
    }

    # Record hoop rebound zone (reduced for more accurate detection)
    hoop_rebound_zone = 5  # Define a smaller buffer zone around the hoop for stricter judgment
    debug_info['rebound_zone'] = {
        'left_boundary': float(rim_x1 - hoop_rebound_zone),
        'right_boundary': float(rim_x2 + hoop_rebound_zone),
        'zone_width': float(hoop_rebound_zone)
    }

    # Check if predicted path crosses the rim area
    is_direct_hit = bool(rim_x1 < predicted_x < rim_x2)
    
    # Check if ball enters rebound zone near the hoop
    is_rebound_hit = bool((rim_x1 - hoop_rebound_zone < predicted_x < rim_x1) or (rim_x2 < predicted_x < rim_x2 + hoop_rebound_zone))
    
    debug_info['shot_analysis'] = {
        'is_direct_hit': bool(is_direct_hit),
        'is_rebound_hit': bool(is_rebound_hit),
        'horizontal_distance_from_center': float(predicted_x - hoop_pos[-1][0][0]),
        'horizontal_distance_from_left_rim': float(predicted_x - rim_x1),
        'horizontal_distance_from_right_rim': float(rim_x2 - predicted_x)
    }
    
    if is_direct_hit:
        debug_info['success_reason'] = "Ball passes directly through the hoop"
        return True
    elif is_rebound_hit:
        debug_info['success_reason'] = "Ball rebounds from the rim edge and goes in"
        return True
    else:
        if predicted_x < rim_x1 - hoop_rebound_zone:
            debug_info['failure_reason'] = "Ball misses from the left side of the hoop"
            debug_info['miss_distance'] = float(rim_x1 - hoop_rebound_zone - predicted_x)
        else:  # predicted_x > rim_x2 + hoop_rebound_zone
            debug_info['failure_reason'] = "Ball misses from the right side of the hoop"
            debug_info['miss_distance'] = float(predicted_x - (rim_x2 + hoop_rebound_zone))
        
        return False


# Detects if the ball is below the net - used to detect shot attempts
def detect_down(ball_pos, hoop_pos):
    """
    检测球是否在篮筐下方（Down区域）
    使用篮筐上边缘的Y值作为分界线
    """
    if not hoop_pos:
        return False
    
    # 篮筐上边缘的Y值
    rim_top_y = hoop_pos[-1][0][1] - 0.5 * hoop_pos[-1][3]
    
    # 球的位置Y值
    ball_y = ball_pos[-1][0][1]
    
    # 如果球在篮筐上边缘下方，则认为是Down区域
    return ball_y > rim_top_y


# Detects if the ball is around the backboard - used to detect shot attempts
def detect_up(ball_pos, hoop_pos):
    x1 = hoop_pos[-1][0][0] - 4 * hoop_pos[-1][2]
    x2 = hoop_pos[-1][0][0] + 4 * hoop_pos[-1][2]
    y1 = hoop_pos[-1][0][1] - 2 * hoop_pos[-1][3]
    y2 = hoop_pos[-1][0][1] - 0.5 * hoop_pos[-1][3]  # ✅ 正确定义

    if x1 < ball_pos[-1][0][0] < x2 and y1 < ball_pos[-1][0][1] < y2:
        return True
    return False


# Checks if center point is near the hoop
def in_hoop_region(center, hoop_pos):
    if len(hoop_pos) < 1:
        return False