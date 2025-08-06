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
    
    # 创建一个字典来快速查找ball_pos中特定帧的数据
    ball_frames = {pos[1]: pos for pos in ball_pos}
    
    # 从hoop_pos的最后一帧开始向前查找，直到找到与篮球数据匹配的帧
    found_matching_frame = False
    
    above_hoop_pos = None
    below_hoop_pos = None
    rim_above = None
    rim_below = None
        # Get first point above rim and first point below rim
    above_point = None
    below_point = None

    for i in reversed(range(len(hoop_pos))):
        hoop_frame = hoop_pos[i][1]
        if hoop_frame in ball_frames:
            # 找到了匹配的帧
            found_matching_frame = True
            hoop = hoop_pos[i]
            ball = ball_frames[hoop_frame]

            rim_above = hoop[0][1] - 0.5 * hoop[3]
            if above_point is None and ball[0][1] < rim_above:
                above_hoop_pos=hoop
                x.append(ball[0][0])
                y.append(ball[0][1])
                above_point = {'x': ball[0][0], 'y': ball[0][1], 'frame':ball[1]}
                debug_info['hoop_info_above_rim'] = {
                    'position': {'x': hoop[0][0], 'y': hoop[0][1]},
                    'width': hoop[2],
                    'height': hoop[3],
                    'rim_height': rim_above,
                    'frame': hoop_frame
                }
                if above_point and below_point :
                    break
            
            rim_below = hoop[0][1] + 0.5 * hoop[3]
            if below_point is None and ball[0][1] > rim_below:
                below_hoop_pos=hoop
                x.append(ball[0][0])
                y.append(ball[0][1])
                below_point = {'x': ball[0][0], 'y': ball[0][1], 'frame': ball[1]}
                 
                debug_info['hoop_info_below_rim'] = {
                    'position': {'x': hoop[0][0], 'y': hoop[0][1]},
                    'width': hoop[2],
                    'height': hoop[3],
                    'rim_height': rim_above,
                    'frame': hoop_frame
                }
                if above_point and below_point :
                    break

    
    # 如果没找到匹配的帧，记录失败原因并返回false
    if not found_matching_frame:
        debug_info['failure_reason'] = "未找到与篮球数据匹配的篮筐帧"
        return False
    

    # 如果没找到同一帧的above_point，记录失败原因并返回false
    if above_point is None:
        debug_info['failure_reason'] = "未找到与篮筐同一帧的above_point数据"
        return False
    
    # 如果没找到同一帧的below_point，记录失败原因并返回false
    if below_point is None:
        debug_info['failure_reason'] = "未找到与篮筐同一帧的below_point数据"
        return False

    debug_info['key_points'] = {
        'above_rim_point': above_point,
        'below_rim_point': below_point
    }
    
    # If not enough points found to create trajectory line
    if len(x) <= 1:
        debug_info['failure_reason'] = "Not enough trajectory points to determine shot result"
        return False

    # Create line from two points
    # m, b = np.polyfit(x, y, 1)
    # average_hoop_pos = None
    
    # predicted_x = ((average_hoop_pos[0][1] - 0.5 * average_hoop_pos[3]) - b) / m
    # rim_x1 = average_hoop_pos[0][0] - 0.4 * average_hoop_pos[2]
    # rim_x2 = average_hoop_pos[0][0] + 0.4 * average_hoop_pos[2]

    # Create line from two points
    m, b = np.polyfit(x, y, 1)
    
    # Calculate average hoop position based on above and below hoop positions
    average_hoop_pos = None
    if above_hoop_pos is not None and below_hoop_pos is not None:
        # Calculate center distance between above and below hoop positions
        center_x1, center_y1 = above_hoop_pos[0][0], above_hoop_pos[0][1]
        center_x2, center_y2 = below_hoop_pos[0][0], below_hoop_pos[0][1]
        
        distance = math.sqrt((center_x2 - center_x1)**2 + (center_y2 - center_y1)**2)
        
        # If distance is less than 5, calculate average position
        if distance < 5:
            avg_center_x = (center_x1 + center_x2) / 2
            avg_center_y = (center_y1 + center_y2) / 2
            avg_width = (above_hoop_pos[2] + below_hoop_pos[2]) / 2
            avg_height = (above_hoop_pos[3] + below_hoop_pos[3]) / 2
            
            # Use the frame from the hoop position (should be the same for both)
            frame = above_hoop_pos[1]
            
            # Create average hoop position
            average_hoop_pos = (np.array([avg_center_x, avg_center_y]), frame, avg_width, avg_height, above_hoop_pos[4])
        else:
            # If distance >= 5, use the latest hoop position (closest frame to current)
            average_hoop_pos = below_hoop_pos
    else:
        # Fallback to whichever hoop position is available
        if above_hoop_pos is not None:
            average_hoop_pos = above_hoop_pos
        elif below_hoop_pos is not None:
            average_hoop_pos = below_hoop_pos
        else:
            # This should not happen based on the validation above
            debug_info['failure_reason'] = "No hoop position available for calculation"
            return False
    
    predicted_x = ((average_hoop_pos[0][1] - 0.5 * average_hoop_pos[3]) - b) / m
    rim_x1 = average_hoop_pos[0][0] - 0.4 * average_hoop_pos[2]
    rim_x2 = average_hoop_pos[0][0] + 0.4 * average_hoop_pos[2]
    
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

    # Record hoop rebound zone
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
        'horizontal_distance_from_center': float(predicted_x - average_hoop_pos[0][0]),
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
def detect_down(ball_data, hoop_data):
    """
    Detect if the ball is below the net
    
    Args:
        ball_data: Ball data tuple (center, frame_count, width, height, conf)
        hoop_data: Hoop data tuple (center, frame_count, width, height, conf)
        
    Returns:
        bool: True if ball is below the net
    """
    # Check if frame counts match
    if ball_data[1] != hoop_data[1]:
        #print(f"DEBUG: detect_down - Frame mismatch. Ball frame: {ball_data[1]}, Hoop frame: {hoop_data[1]}")
        return False
        
    y = hoop_data[0][1] + 0.5 * hoop_data[3]
    result = ball_data[0][1] > y
    #print(f"DEBUG: detect_down - Ball Y: {ball_data[0][1]}, Hoop Y threshold: {y}, Result: {result}")
    return result


# Detects if the ball is around the backboard - used to detect shot attempts
def detect_up(ball_data, hoop_data):
    """
    Detect if the ball is around the backboard
    
    Args:
        ball_data: Ball data tuple (center, frame_count, width, height, conf)
        hoop_data: Hoop data tuple (center, frame_count, width, height, conf)
        
    Returns:
        bool: True if ball is around the backboard
    """
    # Check if frame counts match
    if ball_data[1] != hoop_data[1]:
        #print(f"DEBUG: detect_up - Frame mismatch. Ball frame: {ball_data[1]}, Hoop frame: {hoop_data[1]}")
        return False
        
    x1 = hoop_data[0][0] - 4 * hoop_data[2]
    x2 = hoop_data[0][0] + 4 * hoop_data[2]
    y1 = hoop_data[0][1] - 2 * hoop_data[3]
    y2 = hoop_data[0][1]

    result = x1 < ball_data[0][0] < x2 and y1 < ball_data[0][1] < y2 - 0.5 * hoop_data[3]
    #print(f"DEBUG: detect_up - Ball position: {ball_data[0]}, Hoop position: {hoop_data[0]}, Bounds: x[{x1}, {x2}], y[{y1}, {y2 - 0.5 * hoop_data[3]}], Result: {result}")
    return result


# Checks if center point is near the hoop
def in_hoop_region(center, hoop_data):
    """
    Check if center point is near the hoop
    
    Args:
        center: Center point coordinates (x, y)
        hoop_data: Hoop data tuple (center, frame_count, width, height, conf)
        
    Returns:
        bool: True if center point is near the hoop
    """    
    x = center[0]
    y = center[1]

    x1 = hoop_data[0][0] - 1 * hoop_data[2]
    x2 = hoop_data[0][0] + 1 * hoop_data[2]
    y1 = hoop_data[0][1] - 1 * hoop_data[3]
    y2 = hoop_data[0][1] + 0.5 * hoop_data[3]

    if x1 < x < x2 and y1 < y < y2:
        return True
    return False


# Removes inaccurate data points and selects the best ball from current frame
def select_ball(ball_pos, current_frame_balls, confidence_threshold):
    """
    Clean existing ball positions and select the best ball from current frame detections
    
    Args:
        ball_pos: List of historical ball positions
        current_frame_balls: List of ball detections in current frame
        confidence_threshold: Minimum confidence threshold for ball detection
        
    Returns:
        selected_ball: The selected ball or None if no suitable ball is found
    """
    # Remove points older than 30 frames
    if len(ball_pos) > 0:
        if ball_pos[0][1] < ball_pos[-1][1] - 30:
            ball_pos.pop(0)
    
    # Select the best ball from current frame detections
    selected_ball = None
    
    # Step 1: Filter by confidence
    high_conf_balls = [ball for ball in current_frame_balls if ball['confidence'] > confidence_threshold]
    
    if not high_conf_balls:
        return None
    
    # Step 2: Filter by shape (should be relatively square)
    shaped_balls = []
    for ball in high_conf_balls:
        w = ball['size']['width']
        h = ball['size']['height']
        if not (w * 1.4 < h or h * 1.4 < w):  # Ball should be relatively square
            shaped_balls.append(ball)
    
    if not shaped_balls:
        return None
    
    # Step 3: Filter by movement distance if we have previous ball positions
    if len(ball_pos) > 0 and len(shaped_balls) > 0:
        valid_balls = []
        last_ball = ball_pos[-1]
        last_x, last_y = last_ball[0]
        last_frame = last_ball[1]
        last_w, last_h = last_ball[2], last_ball[3]
        
        max_allowed_dist = 4 * math.sqrt(last_w ** 2 + last_h ** 2)
        
        for ball in shaped_balls:
            curr_x, curr_y = ball['center']
            dist = math.sqrt((curr_x - last_x) ** 2 + (curr_y - last_y) ** 2)
            
            # Check if distance is reasonable (less than max allowed) or if frame gap is large enough
            frame_gap = ball['frame'] - last_frame
            if dist <= max_allowed_dist or frame_gap >= 5:
                valid_balls.append((ball, dist))
    else:
        # If no previous ball positions, all shaped balls are valid
        valid_balls = [(ball, 0) for ball in shaped_balls]
    
    if not valid_balls:
        return None
    
    # Step 4: Select the ball with minimum distance to previous position
    # If no previous position, just take the first valid ball
    if len(ball_pos) > 0:
        selected_ball = min(valid_balls, key=lambda x: x[1])[0]
    else:
        selected_ball = valid_balls[0][0]
    
    return selected_ball


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