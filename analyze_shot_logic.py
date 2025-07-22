#!/usr/bin/env python3
"""
分析shot识别逻辑
Analyze shot detection logic
"""

import json
import argparse

def analyze_shot_detection_logic(shot_log_path, debug_log_path, frame_log_path):
    """分析shot识别的具体逻辑"""
    
    # 加载数据
    with open(shot_log_path, 'r') as f:
        shot_data = json.load(f)
    
    with open(debug_log_path, 'r') as f:
        debug_data = json.load(f)
    
    with open(frame_log_path, 'r') as f:
        frame_data = json.load(f)
    
    print("🏀 SHOT DETECTION LOGIC ANALYSIS")
    print("="*60)
    
    shots = shot_data.get('shots', [])
    detailed_shots = debug_data.get('detailed_shots', [])
    
    print(f"Total shots detected: {len(shots)}")
    print(f"Successful shots: {shot_data.get('successful_shots', 0)}")
    print(f"Shooting accuracy: {shot_data.get('shooting_accuracy', 0)}%")
    
    # 分析每个shot
    for i, shot in enumerate(shots):
        shot_num = i + 1
        frame_idx = shot['frame_index']
        timestamp = shot['timestamp']
        is_successful = shot['is_successful']
        
        print(f"\n📋 SHOT {shot_num} ANALYSIS:")
        print(f"  Frame: {frame_idx} (timestamp: {timestamp:.2f}s)")
        print(f"  Result: {'✅ SUCCESSFUL' if is_successful else '❌ MISSED'}")
        
        # 从debug数据中获取详细信息
        if i < len(detailed_shots):
            debug_shot = detailed_shots[i]
            
            # 球的位置和置信度
            if 'ball_position' in debug_shot:
                ball_pos = debug_shot['ball_position']
                ball_conf = debug_shot.get('ball_confidence', 'N/A')
                print(f"  Ball: position=({ball_pos[0]}, {ball_pos[1]}), confidence={ball_conf}")
            
            # 篮筐位置
            if 'hoop_position' in debug_shot:
                hoop_pos = debug_shot['hoop_position']
                print(f"  Hoop: position=({hoop_pos[0]}, {hoop_pos[1]})")
            
            # 分析轨迹数据
            if 'ball_tracking' in debug_shot:
                ball_tracking = debug_shot['ball_tracking']
                print(f"  Ball tracking: {len(ball_tracking)} points")
                
                if ball_tracking:
                    # 找到up和down的关键帧
                    up_frames = []
                    down_frames = []
                    
                    # 获取篮筐位置用于计算up/down区域
                    if 'hoop_position' in debug_shot:
                        hoop_x, hoop_y = debug_shot['hoop_position']
                        hoop_w, hoop_h = 55, 63  # 假设的篮筐尺寸
                        
                        # up区域: 篮筐上方
                        up_x1 = hoop_x - 4 * hoop_w
                        up_x2 = hoop_x + 4 * hoop_w
                        up_y1 = hoop_y - 2 * hoop_h
                        up_y2 = hoop_y
                        
                        # down区域: 篮筐下方
                        down_y = hoop_y + 0.5 * hoop_h
                        
                        for track in ball_tracking:
                            ball_x = track['position']['x']
                            ball_y = track['position']['y']
                            frame_num = track['frame']
                            
                            # 检查是否在up区域
                            if (up_x1 < ball_x < up_x2 and 
                                up_y1 < ball_y < up_y2 - 0.5 * hoop_h):
                                up_frames.append(frame_num)
                            
                            # 检查是否在down区域
                            if ball_y > down_y:
                                down_frames.append(frame_num)
                        
                        print(f"  Up frames: {up_frames}")
                        print(f"  Down frames: {down_frames}")
                        
                        # 分析shot触发逻辑
                        if up_frames and down_frames:
                            min_up = min(up_frames)
                            min_down = min(down_frames)
                            if min_up < min_down:
                                print(f"  ✅ Shot triggered: ball went from UP (frame {min_up}) to DOWN (frame {min_down})")
                            else:
                                print(f"  ❌ Shot logic issue: DOWN before UP")
                        elif up_frames:
                            print(f"  ⚠️ Ball in UP area only (frames: {up_frames})")
                        elif down_frames:
                            print(f"  ⚠️ Ball in DOWN area only (frames: {down_frames})")
                        else:
                            print(f"  ❓ Ball never in UP or DOWN area")
            
            # 显示成功/失败原因
            if 'result_reason' in debug_shot:
                print(f"  Reason: {debug_shot['result_reason']}")
        
        # 分析对应的frame数据
        if frame_idx < len(frame_data):
            frame_info = frame_data[frame_idx]
            current_balls = frame_info.get('current_detections', {}).get('balls', [])
            current_hoops = frame_info.get('current_detections', {}).get('hoops', [])
            
            print(f"  Frame {frame_idx} detections:")
            print(f"    Current balls: {len(current_balls)}")
            print(f"    Current hoops: {len(current_hoops)}")
            
            if current_balls:
                for j, ball in enumerate(current_balls):
                    print(f"      Ball {j+1}: pos=({ball['center'][0]}, {ball['center'][1]}), conf={ball['confidence']:.3f}")
            
            if current_hoops:
                for j, hoop in enumerate(current_hoops):
                    print(f"      Hoop {j+1}: pos=({hoop['center'][0]}, {hoop['center'][1]}), conf={hoop['confidence']:.3f}")

def analyze_shot_timing_pattern(shot_log_path):
    """分析shot检测的时间模式"""
    
    with open(shot_log_path, 'r') as f:
        shot_data = json.load(f)
    
    shots = shot_data.get('shots', [])
    
    print(f"\n⏱️ SHOT TIMING PATTERN ANALYSIS:")
    print(f"{'Shot':<6} {'Frame':<8} {'Time(s)':<8} {'Interval(s)':<12} {'Interval(frames)':<15}")
    print("-" * 55)
    
    prev_frame = 0
    prev_time = 0
    
    for i, shot in enumerate(shots):
        frame_idx = shot['frame_index']
        timestamp = shot['timestamp']
        
        if i > 0:
            frame_interval = frame_idx - prev_frame
            time_interval = timestamp - prev_time
        else:
            frame_interval = frame_idx
            time_interval = timestamp
        
        print(f"{i+1:<6} {frame_idx:<8} {timestamp:<8.2f} {time_interval:<12.2f} {frame_interval:<15}")
        
        prev_frame = frame_idx
        prev_time = timestamp
    
    # 分析模式
    if len(shots) > 1:
        intervals = []
        for i in range(1, len(shots)):
            interval = shots[i]['frame_index'] - shots[i-1]['frame_index']
            intervals.append(interval)
        
        avg_interval = sum(intervals) / len(intervals)
        print(f"\nAverage interval between shots: {avg_interval:.1f} frames ({avg_interval/30:.2f} seconds)")
        
        # 检查是否有规律性
        if all(abs(interval - 10) <= 2 for interval in intervals):
            print("🔍 Pattern detected: Shots are detected every ~10 frames (every 0.33s)")
            print("   This suggests the shot detection runs every 10 frames as per the code:")
            print("   if self.frame_count % 10 == 0:")

def main():
    parser = argparse.ArgumentParser(description='Analyze shot detection logic')
    parser.add_argument('shot_log', help='Path to shot log JSON file')
    parser.add_argument('debug_log', help='Path to debug log JSON file')
    parser.add_argument('frame_log', help='Path to frame log JSON file')
    
    args = parser.parse_args()
    
    analyze_shot_detection_logic(args.shot_log, args.debug_log, args.frame_log)
    analyze_shot_timing_pattern(args.shot_log)

if __name__ == "__main__":
    main()
