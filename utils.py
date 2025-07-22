import math
import numpy as np
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


def score(ball_pos, hoop_pos, debug_info=None):
    """
    判断投篮是否成功，并收集调试信息
    
    Args:
        ball_pos: 球的位置轨迹
        hoop_pos: 篮筐的位置
        debug_info: 可选的字典，用于存储调试信息
    
    Returns:
        bool: 投篮是否成功
    """
    # 初始化调试信息字典
    if debug_info is None:
        debug_info = {}
    
    x = []
    y = []
    rim_height = hoop_pos[-1][0][1] - 0.5 * hoop_pos[-1][3]
    
    # 记录篮筐信息
    debug_info['hoop_info'] = {
        'position': {'x': hoop_pos[-1][0][0], 'y': hoop_pos[-1][0][1]},
        'width': hoop_pos[-1][2],
        'height': hoop_pos[-1][3],
        'rim_height': rim_height
    }
    
    # 记录球的轨迹点
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
    
    for i in reversed(range(len(ball_pos))):
        if ball_pos[i][0][1] < rim_height:
            x.append(ball_pos[i][0][0])
            y.append(ball_pos[i][0][1])
            above_point = {'x': ball_pos[i][0][0], 'y': ball_pos[i][0][1], 'frame': ball_pos[i][1]}
            if i + 1 < len(ball_pos):
                x.append(ball_pos[i + 1][0][0])
                y.append(ball_pos[i + 1][0][1])
                below_point = {'x': ball_pos[i + 1][0][0], 'y': ball_pos[i + 1][0][1], 'frame': ball_pos[i + 1][1]}
            break
    
    debug_info['key_points'] = {
        'above_rim_point': above_point,
        'below_rim_point': below_point
    }
    
    # 如果没有找到足够的点来创建轨迹线
    if len(x) <= 1:
        debug_info['failure_reason'] = "没有足够的轨迹点来判断投篮结果"
        return False

    # Create line from two points
    m, b = np.polyfit(x, y, 1)
    predicted_x = ((hoop_pos[-1][0][1] - 0.5 * hoop_pos[-1][3]) - b) / m
    rim_x1 = hoop_pos[-1][0][0] - 0.4 * hoop_pos[-1][2]
    rim_x2 = hoop_pos[-1][0][0] + 0.4 * hoop_pos[-1][2]
    
    # 记录轨迹线和预测信息
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

    # 记录篮筐反弹区域
    hoop_rebound_zone = 10  # Define a buffer zone around the hoop
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
        debug_info['success_reason'] = "球直接穿过篮筐"
        return True
    elif is_rebound_hit:
        debug_info['success_reason'] = "球从篮筐边缘反弹进入"
        return True
    else:
        if predicted_x < rim_x1 - hoop_rebound_zone:
            debug_info['failure_reason'] = "球从篮筐左侧错过"
            debug_info['miss_distance'] = float(rim_x1 - hoop_rebound_zone - predicted_x)
        else:  # predicted_x > rim_x2 + hoop_rebound_zone
            debug_info['failure_reason'] = "球从篮筐右侧错过"
            debug_info['miss_distance'] = float(predicted_x - (rim_x2 + hoop_rebound_zone))
        
        return False


# Detects if the ball is below the net - used to detect shot attempts
def detect_down(ball_pos, hoop_pos):
    y = hoop_pos[-1][0][1] + 0.5 * hoop_pos[-1][3]
    if ball_pos[-1][0][1] > y:
        return True
    return False


# Detects if the ball is around the backboard - used to detect shot attempts
def detect_up(ball_pos, hoop_pos):
    x1 = hoop_pos[-1][0][0] - 4 * hoop_pos[-1][2]
    x2 = hoop_pos[-1][0][0] + 4 * hoop_pos[-1][2]
    y1 = hoop_pos[-1][0][1] - 2 * hoop_pos[-1][3]
    y2 = hoop_pos[-1][0][1]

    if x1 < ball_pos[-1][0][0] < x2 and y1 < ball_pos[-1][0][1] < y2 - 0.5 * hoop_pos[-1][3]:
        return True
    return False


# Checks if center point is near the hoop
def in_hoop_region(center, hoop_pos):
    if len(hoop_pos) < 1:
        return False
    x = center[0]
    y = center[1]

    x1 = hoop_pos[-1][0][0] - 1 * hoop_pos[-1][2]
    x2 = hoop_pos[-1][0][0] + 1 * hoop_pos[-1][2]
    y1 = hoop_pos[-1][0][1] - 1 * hoop_pos[-1][3]
    y2 = hoop_pos[-1][0][1] + 0.5 * hoop_pos[-1][3]

    if x1 < x < x2 and y1 < y < y2:
        return True
    return False


# Removes inaccurate data points
def clean_ball_pos(ball_pos, frame_count):
    # Removes inaccurate ball size to prevent jumping to wrong ball
    if len(ball_pos) > 1:
        # Width and Height
        w1 = ball_pos[-2][2]
        h1 = ball_pos[-2][3]
        w2 = ball_pos[-1][2]
        h2 = ball_pos[-1][3]

        # X and Y coordinates
        x1 = ball_pos[-2][0][0]
        y1 = ball_pos[-2][0][1]
        x2 = ball_pos[-1][0][0]
        y2 = ball_pos[-1][0][1]

        # Frame count
        f1 = ball_pos[-2][1]
        f2 = ball_pos[-1][1]
        f_dif = f2 - f1

        dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        max_dist = 4 * math.sqrt((w1) ** 2 + (h1) ** 2)

        # Ball should not move a 4x its diameter within 5 frames
        if (dist > max_dist) and (f_dif < 5):
            ball_pos.pop()

        # Ball should be relatively square
        elif (w2*1.4 < h2) or (h2*1.4 < w2):
            ball_pos.pop()

    # Remove points older than 30 frames
    if len(ball_pos) > 0:
        if frame_count - ball_pos[0][1] > 30:
            ball_pos.pop(0)

    return ball_pos


def clean_hoop_pos(hoop_pos):
    # Prevents jumping from one hoop to another
    if len(hoop_pos) > 1:
        x1 = hoop_pos[-2][0][0]
        y1 = hoop_pos[-2][0][1]
        x2 = hoop_pos[-1][0][0]
        y2 = hoop_pos[-1][0][1]

        w1 = hoop_pos[-2][2]
        h1 = hoop_pos[-2][3]
        w2 = hoop_pos[-1][2]
        h2 = hoop_pos[-1][3]

        f1 = hoop_pos[-2][1]
        f2 = hoop_pos[-1][1]

        f_dif = f2-f1

        dist = math.sqrt((x2-x1)**2 + (y2-y1)**2)

        max_dist = 0.5 * math.sqrt(w1 ** 2 + h1 ** 2)

        # Hoop should not move 0.5x its diameter within 5 frames
        if dist > max_dist and f_dif < 5:
            hoop_pos.pop()

        # Hoop should be relatively square
        if (w2*1.3 < h2) or (h2*1.3 < w2):
            hoop_pos.pop()

    # Remove old points
    if len(hoop_pos) > 25:
        hoop_pos.pop(0)

    return hoop_pos