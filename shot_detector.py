# Avi Shah - Basketball Shot Detector/Tracker - July 2023

from ultralytics import YOLO
import cv2
import numpy as np
import math
import time
import logging
import json
from collections import deque
import cvzone
from scenedetect import detect, ContentDetector
from tqdm import tqdm

from utils import score, in_hoop_region, clean_hoop_pos, clean_ball_pos, get_device
from datetime import datetime
import logging
import os

from logging_utils import DebugLogger, ShotLogger
from object_class_mapper import object_class_mapper
from ball_tracker import BallTracker
from hoop_tracker import HoopTracker
from shot_analyzer import ShotAnalyzer


class ShotDetector:
    def __init__(self, input_video="video_test_5.mp4", output_video=None, ball_model_path="yolov8m.pt", hoop_model_path=None, person_model_path=None, use_shared_model=True, min_ball_area=400, enable_person_detection=False, model_config=None):
        import os
        # Load models for optimal detection
        self.use_shared_model = use_shared_model

        self.model_config = model_config
        self.enable_person_detection = enable_person_detection

        # Load main detection model (YOLOv8m for sports ball and person)
        self.ball_model_path = ball_model_path
        self.ball_model = YOLO(ball_model_path)
        print(f"🏀 Loaded main model: {ball_model_path}")
        
        # Load hoop detection model (use ball model if not specified)
        self.hoop_model_path = hoop_model_path if hoop_model_path else ball_model_path
        self.hoop_model = YOLO(self.hoop_model_path)
        print(f"🏀 Loaded hoop model: {self.hoop_model_path}")
        
        # Initialize model classes
        self.ball_model_classes = self.ball_model.names if hasattr(self.ball_model, 'names') else {0: "Basketball"}
        self.hoop_model_classes = self.hoop_model.names if hasattr(self.hoop_model, 'names') else {0: "Basketball", 1: "Basketball Hoop"}
        
        # Print ball model classes
        ball_classes_list = list(self.ball_model_classes.values())
        if 'sports ball' in [cls.lower() for cls in ball_classes_list]:
            print(f"📋 Ball model classes: {len(self.ball_model_classes)} classes (including 'sports ball')")
        else:
            print(f"📋 Ball model classes: {len(self.ball_model_classes)} classes: {ball_classes_list}")
        
        # Print hoop model classes
        print(f"📋 Hoop model classes: {list(self.hoop_model_classes.values())}")
        
        # Print active hoop detection classes
        active_hoop_classes = object_class_mapper.get_class_names_for_type('hoop')
        print(f"🎯 Active hoop detection classes: {active_hoop_classes}")

        # Person detection: only load if enabled
        if enable_person_detection:
            if use_shared_model:
                self.person_model = self.ball_model  # Share the same model
                self.person_model_path = ball_model_path
                self.person_model_classes = self.ball_model.names if hasattr(self.ball_model, 'names') else {0: "person"}
                print(f"👤 Using shared model for person detection: {ball_model_path}")
            else:
                # Use separate person model if specified
                self.person_model_path = person_model_path or "yolov8n.pt"
                self.person_model = YOLO(self.person_model_path)
                self.person_model_classes = self.person_model.names if hasattr(self.person_model, 'names') else {0: "person"}
                print(f"👤 Loaded separate person model: {self.person_model_path}")
                
            # 打印人物检测类别
            person_classes = [cls for cls in self.person_model_classes.values() 
                             if 'person' in cls.lower()]
            print(f"📋 Person model classes: {len(self.person_model_classes)} classes")
            print(f"👤 Active person detection classes: {person_classes}")
        else:
            self.person_model = None
            self.person_model_path = None
            self.person_model_classes = {0: "person"}  # 默认值，即使未启用也提供
            print(f"👤 Person detection disabled")

        # For backward compatibility, set primary model as ball model
        self.model = self.ball_model
        self.model_path = ball_model_path
        self.input_video = input_video

        # --- Output video filename logic ---
        if output_video is None:
            video_name = os.path.splitext(os.path.basename(input_video))[0]
            if not ball_model_path:
                raise ValueError("Model name (ball_model_path) must be provided for output video file naming.")
            model_name = os.path.splitext(os.path.basename(ball_model_path))[0]
            timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            # 安全处理文件名，移除不安全字符
            video_name = ''.join(c for c in video_name if c.isalnum() or c in '._- ')
            model_name = ''.join(c for c in model_name if c.isalnum() or c in '._- ')
            output_video = f"{video_name}_{model_name}_output_{timestamp}.mp4"
        self.output_video = output_video
        self.video_writer = None

        # Initialize two loggers - one for frame data and one for shot data
        self.frame_logger = ShotLogger(input_file=self.input_video, model_path=self.ball_model_path, log_type="frame")
        self.shot_logger = ShotLogger(input_file=self.input_video, model_path=self.ball_model_path, log_type="shot")
        # 关键：让 ShotDetector 直接引用 ShotLogger 的 debug_logger
        self.debug_logger = DebugLogger(debug_log_file=os.path.join('logs', 'console_output.log'), input_file=self.input_video, model_path=self.ball_model_path, log_type="debug")
        self.debug_logger.debug_file_only("[ShotDetector] debug_logger now references ShotLogger's initialized logger.")

        # 设置最小球面积阈值
        self.min_ball_area = min_ball_area  # 使用传入的参数值

        # Initialize tracking components
        self.ball_tracker = BallTracker(min_ball_area=self.min_ball_area, debug_logger=self.debug_logger)
        self.hoop_tracker = HoopTracker(debug_logger=self.debug_logger)
        self.shot_analyzer = ShotAnalyzer(debug_logger=self.debug_logger)

        if hasattr(self.person_model, 'names'):
            self.person_model_classes = self.person_model.names
            # Filter and show only person-related classes
            person_classes = [cls for cls in self.person_model_classes.values()
                             if 'person' in cls.lower()]
            print(f"📋 Person model classes: {len(self.person_model_classes)} classes")
            print(f"👤 Active person detection classes: {person_classes}")
        else:
            self.person_model_classes = {0: "person"}
        self.device = get_device()
        # Uncomment line below to use webcam (I streamed to my iPhone using Iriun Webcam)
        # self.cap = cv2.VideoCapture(0)

        # Use video from input parameter
        self.cap = cv2.VideoCapture(input_video)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.frame_count = 0  # 添加frame_count初始化
        
        # Initialize tracking variables
        self.person_pos = []
        self.selected_ball = None
        self.selected_hoop = None
        
        # Initialize score tracking
        self.makes = 0
        self.attempts = 0
        
        # Initialize overlay properties
        self.overlay_color = (0, 0, 255)  # 默认红色
        self.fade_counter = 0
        self.fade_frames = 30  # 默认淡出帧数

        # 使用PySceneDetect预先检测所有场景变化帧
        self.scene_change_frames = self.detect_scene_changes_pyscenedetect(input_video)
        self.debug_logger.info(f"🎬 预先检测到的场景变化帧: {self.scene_change_frames}")
        
        # 初始化篮网扰动检测相关变量
        self.ball_in_hoop_area = False
        self.ball_entered_hoop_frame = 0
        self.net_reference_frame = None
        self._net_disturbance_detected = False
        
        # 初始化前一帧变量
        self.prev_frame = None

    def detect_scene_changes_pyscenedetect(self, video_path):
        """
        使用PySceneDetect库检测视频中的场景变化
        
        Args:
            video_path: 视频文件路径
            
        Returns:
            list: 包含场景变化帧号的列表
        """
        try:
            # 使用ContentDetector检测场景变化
            scene_list = detect(video_path, ContentDetector())
            # 提取场景变化的帧号
            scene_frames = []
            for scene in scene_list:
                start_time = scene[0]
                # 将时间戳转换为帧号
                frame_number = int(start_time.get_frames())
                scene_frames.append(frame_number)
            
            self.debug_logger.info(f"🔍 使用PySceneDetect检测到场景变化帧: {scene_frames}")
            return scene_frames
        except Exception as e:
            self.debug_logger.error(f"❌ PySceneDetect检测失败: {str(e)}")
            return []

    def clean_motion(self):
        """
        Clean up motion data by removing inaccurate data points
        """
        # Clean ball position data
        self.ball_tracker.clean_motion(self.frame_count)
        
        # Clean hoop position data
        self.hoop_tracker.clean_motion()
        
        # Clean person position data if person detection is enabled
        if self.enable_person_detection and len(self.person_pos) > 30:
            self.person_pos.pop(0)

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
            # 优先使用缓存的帧高度
            if hasattr(self, 'frame_height') and self.frame_height > 0:
                frame_height = self.frame_height
            elif hasattr(self, 'frame') and self.frame is not None:
                frame_height = self.frame.shape[0]
            else:
                frame_height = 1080  # fallback if frame not available

        center_y = ball_data['center'][1]

        # Position reasonableness check
        if center_y > frame_height * 1.1:  # Allow 10% margin above typical height
            self.debug_logger.warning(f"Ball Y position {center_y} exceeds reasonable limit {frame_height * 1.1}")
            return False

        # Additional checks can be added here
        # e.g., trajectory continuity, X position bounds, etc.

        return True
        
    def predict_ball_position_from_trajectory(self, current_frame):
        """
        预测当前帧中球的位置，基于历史轨迹数据拟合
        
        Args:
            current_frame: 当前帧号
            
        Returns:
            tuple: 预测的球位置 (x, y) 或 None（如果历史数据不足）
        """
        return self.ball_tracker.predict_ball_position_from_trajectory(current_frame)

    def is_ball_in_hoop_area(self, ball_pos, hoop_pos):
        """
        Check if ball position is within the hoop area
        
        Args:
            ball_pos: Ball position (x, y)
            hoop_pos: Hoop position data
            
        Returns:
            bool: True if ball is in hoop area
        """
        return self.hoop_tracker.is_ball_in_hoop_area(ball_pos, hoop_pos)

    def detect_net_disturbance(self, prev_frame, curr_frame, hoop_region):
        """
        Compare hoop regions between frames to detect net disturbance
        
        Args:
            prev_frame: Previous frame image
            curr_frame: Current frame image
            hoop_region: (x1, y1, x2, y2) coordinates of hoop region
            
        Returns:
            float: Disturbance level (higher means more disturbance)
        """
        # 安全检查：确保输入帧有效
        if prev_frame is None or curr_frame is None:
            return 0.0
            
        # 获取区域坐标
        x1, y1, x2, y2 = hoop_region
        
        # 确保坐标是整数
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # 确保坐标在帧边界内
        height, width = curr_frame.shape[:2]
        x1 = max(0, min(x1, width-1))
        y1 = max(0, min(y1, height-1))
        x2 = max(0, min(x2, width-1))
        y2 = max(0, min(y2, height-1))
        
        # 检查区域是否有效
        if x2 <= x1 or y2 <= y1:
            return 0.0
            
        # 提取两帧中的篮筐区域
        prev_region = prev_frame[y1:y2, x1:x2]
        curr_region = curr_frame[y1:y2, x1:x2]
        
        # 转换为灰度图
        try:
            prev_gray = cv2.cvtColor(prev_region, cv2.COLOR_BGR2GRAY)
            curr_gray = cv2.cvtColor(curr_region, cv2.COLOR_BGR2GRAY)
        except cv2.error:
            # 处理颜色转换错误
            return 0.0
        
        # 计算绝对差异
        diff = cv2.absdiff(prev_gray, curr_gray)
        
        # 应用阈值突出显著变化
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        
        # 计算变化像素的比例
        changed_pixels = np.count_nonzero(thresh)
        total_pixels = thresh.shape[0] * thresh.shape[1]
        
        if total_pixels == 0:
            return 0.0
            
        disturbance_level = changed_pixels / total_pixels
        return disturbance_level

    def check_hoop_disturbance_shot(self):
        """
        Check for successful shot based on net disturbance when ball is not detected
        but was previously moving towards the hoop.
        
        Sequence:
        1. Check if no valid ball detected in current frame (no selected ball)
        2. Check if there was an UP state before  
        3. Check if the hoop is at the edge of the video
        4. Predict ball position using trajectory fitting (only when no selected ball)
        5. If predicted position is in hoop, compare hoop region between frames for net disturbance
        6. If disturbance is detected, count as successful shot
        
        Returns:
            bool: True if a successful shot is detected based on net disturbance
        """
        # 1. Check if no valid ball detected in current frame (no selected ball)
        if self.selected_ball is not None:
            self.debug_logger.debug_file_only("当前帧检测到有效球，跳过篮网扰动检测")
            return False
            
        # 2. Check if there was an UP state before
        if not self.shot_analyzer.up:
            self.debug_logger.debug_file_only("之前没有检测到UP状态，跳过篮网扰动检测")
            return False
            
        # 3. Check if the hoop is at the edge of the video
        if not self.hoop_tracker.hoop_pos:
            self.debug_logger.debug_file_only("未检测到篮筐位置，无法进行篮网扰动检测")
            return False
            
        current_hoop = self.hoop_tracker.hoop_pos[-1]
        hoop_center = current_hoop[0]
        hoop_width = current_hoop[2]
        
        # 优先使用缓存的帧尺寸
        frame_width = self.frame_width or (self.frame.shape[1] if self.frame is not None else 0)
        frame_height = self.frame_height or (self.frame.shape[0] if self.frame is not None else 0)
        
        if frame_width == 0 or frame_height == 0:
            self.debug_logger.debug_file_only("无法获取帧尺寸，跳过篮网扰动检测")
            return False
        
        # Define edge threshold (15% of frame dimensions from edges)
        edge_threshold_x = 0.15 * frame_width
        edge_threshold_y = 0.15 * frame_height
        
        # Check if hoop is near the edge of the video
        is_hoop_at_edge = (hoop_center[0] < edge_threshold_x) or \
                         (hoop_center[0] > frame_width - edge_threshold_x) or \
                         (hoop_center[1] < edge_threshold_y) or \
                         (hoop_center[1] > frame_height - edge_threshold_y)
                         
        if not is_hoop_at_edge:
            self.debug_logger.debug_file_only(f"篮筐不在视频边缘，hoop_center=({hoop_center[0]}, {hoop_center[1]}), frame_size=({frame_width}, {frame_height})")
            return False
            
        self.debug_logger.debug_file_only(f"篮筐在视频边缘，进行篮网扰动检测: hoop_center=({hoop_center[0]}, {hoop_center[1]}), frame_size=({frame_width}, {frame_height})")

        # 4. Use selected ball if available, otherwise predict ball position using trajectory fitting
        ball_position = None
        if self.selected_ball and self.selected_ball[1] == self.frame_count:
            # Use selected ball position if available in current frame
            ball_position = self.selected_ball[0]
            self.debug_logger.debug_file_only(f"使用选中的球位置: {ball_position}")
        else:
            # 4. Predict ball position using trajectory fitting (only when no selected ball)
            ball_position = self.predict_ball_position_from_trajectory(self.frame_count)
        
        if not ball_position:
            self.debug_logger.debug_file_only("无法预测球的位置，跳过篮网扰动检测")
            return False
            
        # 5. Check if ball position is in hoop area
        is_ball_in_hoop_now = self.is_ball_in_hoop_area(ball_position, current_hoop)
        
        # Track ball entering and leaving hoop area for delayed net disturbance detection
        if is_ball_in_hoop_now and not self.ball_in_hoop_area:
            # Ball just entered hoop area
            self.ball_in_hoop_area = True
            self.ball_entered_hoop_frame = self.frame_count
            # Save reference frame (frame before ball entered)
            if self.prev_frame is not None:
                self.net_reference_frame = self.prev_frame.copy()
            self.debug_logger.debug_file_only(f"球进入篮筐区域，帧号: {self.frame_count}")
            
        elif not is_ball_in_hoop_now and self.ball_in_hoop_area:
            # Ball just left hoop area
            self.ball_in_hoop_area = False
            frames_since_entered = self.frame_count - self.ball_entered_hoop_frame
            
            # Only check for net disturbance if ball was in hoop area for a reasonable time
            # and we have a reference frame
            if frames_since_entered >= 3 and self.net_reference_frame is not None:
                self.debug_logger.debug_file_only(f"球离开篮筐区域，进入时帧号: {self.ball_entered_hoop_frame}, 离开时帧号: {self.frame_count}")
                
                # Define hoop region for analysis (focus on the net area)
                x1 = hoop_center[0] - 0.7 * hoop_width
                y1 = hoop_center[1] - 0.2 * current_hoop[3]  # Just below the rim
                x2 = hoop_center[0] + 0.7 * hoop_width
                y2 = hoop_center[1] + 1.5 * current_hoop[3]  # Extend down to capture net movement
                hoop_region = (x1, y1, x2, y2)
                
                # Compare hoop region between reference frame and current frame for net disturbance
                disturbance = self.detect_net_disturbance(self.net_reference_frame, self.frame, hoop_region)
                
                self.debug_logger.debug_file_only(f"篮网扰动检测结果: 扰动水平={disturbance:.4f}")
                
                # Clear reference frame
                self.net_reference_frame = None
                
                # If significant disturbance detected, consider it a successful shot
                if disturbance > 0.01:  # 1% of pixels changed
                    self.debug_logger.info(f"✅ 检测到篮网扰动，判断为进球! 扶动水平: {disturbance:.4f}")
                    # Set a flag to indicate this shot was detected via net disturbance
                    self._net_disturbance_detected = True
                    return True
            else:
                # Clear reference frame if not enough frames passed
                self.net_reference_frame = None
                
        elif is_ball_in_hoop_now:
            # Ball still in hoop area, continue tracking
            self.debug_logger.debug_file_only(f"球仍在篮筐区域内，已持续 {self.frame_count - self.ball_entered_hoop_frame + 1} 帧")
            
        # Update state if ball is in hoop area now
        self.ball_in_hoop_area = is_ball_in_hoop_now
        
        return False

    def detect_scene_change(self, prev_frame, curr_frame, threshold=0.7):
        """
        使用PySceneDetect替代此方法
        """
        return False

    def select_best_detections_for_frame(self, current_frame_balls, current_frame_hoops):
        """
        Select the best ball and hoop detections from current frame for UP/DOWN analysis

        Args:
            current_frame_balls: List of ball detections in current frame
            current_frame_hoops: List of hoop detections in current frame

        Returns:
            tuple: (best_ball, best_hoop) or (None, None) if not both available
        """
        if not current_frame_balls or not current_frame_hoops:
            return None, None

        # Filter high-quality detections - INCREASED ball confidence threshold
        quality_balls = [ball for ball in current_frame_balls
                        if ball['confidence'] >= 0.4 and ball.get('area', 0) >= self.min_ball_area
                        and self.is_reasonable_ball_position(ball)]  # Added position check
        quality_hoops = [hoop for hoop in current_frame_hoops
                        if hoop['confidence'] >= 0.4]

        if not quality_balls or not quality_hoops:
            return None, None
            
        # 如果有选中的球，则优先使用选中的球，否则使用轨迹拟合进行筛选
        if self.selected_ball:
            # 使用选中的球进行匹配
            selected_ball_center = self.selected_ball[0]
            selected_ball_frame = self.selected_ball[1]
            
            # 查找与选中球最接近的当前帧球
            best_ball = None
            min_distance = float('inf')
            
            for ball in quality_balls:
                ball_center = ball['center']
                # 计算与选中球的位置差异
                distance = ((ball_center[0] - selected_ball_center[0])**2 + 
                           (ball_center[1] - selected_ball_center[1])**2)**0.5
                if distance < min_distance:
                    min_distance = distance
                    best_ball = ball
                    
            self.debug_logger.debug_file_only(f"使用选中的球进行匹配，最小距离: {min_distance:.1f}px")
        else:
            # 尝试使用轨迹拟合进行筛选（对每个球都进行检验）
            predicted_position = self.predict_ball_position_from_trajectory(self.frame_count)
            
            if predicted_position and len(self.ball_tracker.ball_pos) >= 3:  # 只有在有足够历史点时才使用轨迹预测
                # 设置基础偏差阈值（像素）
                base_deviation_threshold = 50  # 基础阈值
                
                # 获取拟合数据的帧序列最大值
                recent_history = self.ball_tracker.ball_pos[-10:]
                frames = [pos[1] for pos in recent_history]
                max_history_frame = max(frames) if frames else self.frame_count
                
                # 计算当前帧与拟合数据帧序列最大值的差值，并动态调整阈值
                frame_diff = abs(self.frame_count - max_history_frame)
                # 每帧差异增加基础阈值的比例
                deviation_threshold = base_deviation_threshold * (1 + frame_diff)
                
                self.debug_logger.debug_file_only(f"动态偏差阈值: {deviation_threshold:.1f}px (基础阈值: {base_deviation_threshold}px, 帧差值: {frame_diff})")
                
                # 计算每个球与预测位置的偏差
                for ball in quality_balls:
                    ball_center = ball['center']
                    deviation = ((ball_center[0] - predicted_position[0])**2 + 
                                (ball_center[1] - predicted_position[1])**2)**0.5
                    ball['trajectory_deviation'] = deviation
                    
                    self.debug_logger.debug_file_only(f"球 ({ball_center[0]:.1f}, {ball_center[1]:.1f}) 与预测位置偏差: {deviation:.1f}px")
                
                # 过滤掉偏差超过阈值的球
                trajectory_filtered_balls = [ball for ball in quality_balls 
                                           if ball.get('trajectory_deviation', float('inf')) <= deviation_threshold]
                
                if trajectory_filtered_balls:
                    self.debug_logger.debug_file_only(f"轨迹筛选后剩余 {len(trajectory_filtered_balls)}/{len(quality_balls)} 个球")
                    # 从轨迹筛选后的球中选择置信度最高的
                    best_ball = max(trajectory_filtered_balls, key=lambda x: x['confidence'])
                    self.debug_logger.debug_file_only(f"选择基于轨迹筛选的最佳球: 置信度={best_ball['confidence']:.2f}, 偏差={best_ball.get('trajectory_deviation', 'N/A'):.1f}px")
                else:
                    # 如果所有球都被轨迹筛选过滤掉，则从所有高质量球中选择置信度最高的
                    self.debug_logger.debug_file_only(f"所有球都超出轨迹偏差阈值，使用置信度最高的球")
                    self.debug_logger.debug_file_only(f"最大偏差: {max([ball.get('trajectory_deviation', float('inf')) for ball in quality_balls]):.1f}px, 动态阈值: {deviation_threshold:.1f}px (基础阈值: {base_deviation_threshold}px, 帧差值: {frame_diff})")
                    best_ball = max(quality_balls, key=lambda x: x['confidence'])
            else:
                # 无法进行轨迹预测时，使用置信度最高的球
                self.debug_logger.debug_file_only(f"无法进行轨迹预测，使用置信度最高的球")
                best_ball = max(quality_balls, key=lambda x: x['confidence'])

        # 选择置信度最高的篮筐
        best_hoop = max(quality_hoops, key=lambda x: x['confidence'])

        return best_ball, best_hoop

    def filter_overlapping_persons(self, person_detections):
        """
        Filter overlapping person detections, keeping the one with larger height (full body)

        Args:
            person_detections: List of person detection dictionaries

        Returns:
            List of filtered person detections
        """
        if len(person_detections) <= 1:
            return person_detections

        filtered = []
        used_indices = set()

        for i, detection1 in enumerate(person_detections):
            if i in used_indices:
                continue

            # Check for overlaps with other detections
            overlapping_detections = [detection1]
            overlapping_indices = [i]

            for j, detection2 in enumerate(person_detections):
                if i != j and j not in used_indices:
                    # Calculate overlap with improved algorithm
                    if self.calculate_person_overlap(detection1, detection2) > 0.2:  # 20% overlap threshold (lowered)
                        overlapping_detections.append(detection2)
                        overlapping_indices.append(j)

            # If there are overlapping detections, choose the best one
            if len(overlapping_detections) > 1:
                # Prefer detection with larger height (more likely to be full body)
                # Also consider confidence as secondary factor
                best_detection = max(overlapping_detections,
                                   key=lambda d: (d["size"]["height"], d["confidence"]))
                filtered.append(best_detection)

                # Mark all overlapping indices as used
                used_indices.update(overlapping_indices)
            else:
                # No overlap, keep the detection
                filtered.append(detection1)
                used_indices.add(i)

        return filtered

    def calculate_person_overlap(self, detection1, detection2):
        """
        Calculate overlap ratio between two person detections using improved logic

        Args:
            detection1, detection2: Person detection dictionaries with bbox

        Returns:
            float: Overlap ratio (0-1)
        """
        bbox1 = detection1["bbox"]  # [x1, y1, x2, y2]
        bbox2 = detection2["bbox"]
        center1 = detection1["center"]
        center2 = detection2["center"]

        # Calculate intersection
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])

        if x2 <= x1 or y2 <= y1:
            return 0.0  # No overlap

        intersection_area = (x2 - x1) * (y2 - y1)

        # Calculate areas
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

        # Use intersection over smaller area (more sensitive to partial overlaps)
        smaller_area = min(area1, area2)
        overlap_ratio = intersection_area / smaller_area if smaller_area > 0 else 0.0

        # Additional check: if centers are close and one bbox contains significant part of the other
        center_distance = ((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)**0.5
        avg_width = (detection1["size"]["width"] + detection2["size"]["width"]) / 2

        # If centers are close (within average width) and there's any overlap, consider them overlapping
        if center_distance < avg_width and overlap_ratio > 0.1:
            return max(overlap_ratio, 0.5)  # Boost overlap score for close detections

        return overlap_ratio

    def process_frame_detections(self, current_frame_balls, current_frame_hoops):
        """
        Process current frame detections and update selected values for UP/DOWN analysis
        Only performs UP/DOWN detection when both ball and hoop are detected in same frame

        Args:
            current_frame_balls: List of ball detections in current frame
            current_frame_hoops: List of hoop detections in current frame
        """
        # 🔧 FIX: Store current frame detections for visualization
        self.current_frame_balls = current_frame_balls
        self.current_frame_hoops = current_frame_hoops

        # Store current frame as previous for next iteration (确保在处理前保存)
        if self.frame is not None:
            self.prev_frame = self.frame.copy()

        self.debug_logger.debug_file_only(f"🔥 FORCE DEBUG: process_frame_detections called for frame {self.frame_count}")
        self.debug_logger.debug_file_only(f"🔥 Input: {len(current_frame_balls)} balls, {len(current_frame_hoops)} hoops")

        # Check for significant hoop position/size changes (possible video cut)
        if len(self.hoop_tracker.hoop_pos) > 1 and len(current_frame_hoops) > 0:
            last_hoop = self.hoop_tracker.hoop_pos[-1]
            current_hoop = max(current_frame_hoops, key=lambda x: x['confidence'])
            
            # Calculate position and size differences
            pos_diff = math.sqrt((last_hoop[0][0] - current_hoop['center'][0])**2 + 
                                (last_hoop[0][1] - current_hoop['center'][1])**2)
            size_diff = abs(last_hoop[2] - current_hoop['size']['width']) + \
                       abs(last_hoop[3] - current_hoop['size']['height'])
            
            # 更严格的阈值设置，避免误报
            # 只有当位置变化超过画面宽度的25%且大小变化显著时才认为是视频剪辑
            frame_width = self.frame_width or (self.frame.shape[1] if self.frame is not None else 1920)
            frame_height = self.frame_height or (self.frame.shape[0] if self.frame is not None else 1080)
            pos_threshold = 0.25 * math.sqrt(frame_width**2 + frame_height**2)  # 对角线长度的25%
            size_threshold = 1.0 * (last_hoop[2] + last_hoop[3])  # 更严格的大小变化阈值（100%）
            
            # 添加额外的条件：只有当置信度差异不大时才考虑是剪辑
            conf_diff = abs(last_hoop[4] - current_hoop['confidence'])
            
            # 只有当位置和大小都发生剧烈变化，且置信度变化不大时，才认为是视频剪辑
            if pos_diff > pos_threshold and size_diff > size_threshold and conf_diff < 0.2:
                self.debug_logger.warning(f"⚠️ Frame {self.frame_count}: 检测到篮筐位置/大小显著变化 (位置差: {pos_diff:.1f}px > {pos_threshold:.1f}px 且大小差: {size_diff:.1f}px > {size_threshold:.1f}px)，可能是视频剪辑，重置跟踪数据")
                # 记录视频剪辑事件
                self.frame_logger.log_video_cut(self.frame_count)
                self.ball_tracker.reset_tracking()
                self.hoop_tracker.reset_tracking()
                self.person_pos = []
                self.shot_analyzer.reset_shot_detection()
                return
        
        # Select best detections from current frame
        selected_ball_data, selected_hoop_data = self.select_best_detections_for_frame(
            current_frame_balls, current_frame_hoops
        )

        self.debug_logger.debug(f"🔥 Selected: ball={bool(selected_ball_data)}, hoop={bool(selected_hoop_data)}")

        if selected_ball_data and selected_hoop_data:
            # 🔧 CRITICAL FIX: Apply same filtering logic as select_best_detections_for_frame
            # Ensure ball meets quality requirements before UP/DOWN detection
            if (selected_ball_data['confidence'] < 0.4 or
                selected_ball_data.get('area', 0) < self.min_ball_area or
                not self.is_reasonable_ball_position(selected_ball_data)):
                self.debug_logger.warning(f"🚫 Frame {self.frame_count}: Ball filtered out in UP/DOWN detection")
                self.debug_logger.warning(f"   Ball conf={selected_ball_data['confidence']:.2f}, pos={selected_ball_data['center']}")
                selected_ball_data = None
                selected_hoop_data = None

        if selected_ball_data and selected_hoop_data:
            # Convert to trajectory format (ensuring same frame)
            self.selected_ball = (
                (selected_ball_data['center'][0], selected_ball_data['center'][1]),
                self.frame_count,  # Same frame
                selected_ball_data['size']['width'],
                selected_ball_data['size']['height'],
                selected_ball_data['confidence']
            )

            self.selected_hoop = (
                (selected_hoop_data['center'][0], selected_hoop_data['center'][1]),
                self.frame_count,  # Same frame
                selected_hoop_data['size']['width'],
                selected_hoop_data['size']['height'],
                selected_hoop_data['confidence']
            )

            # Debug: Verify frame numbers
            self.debug_logger.debug(f"🔍 Frame {self.frame_count}: Creating selected data")
            self.debug_logger.debug(f"  Selected ball frame: {self.selected_ball[1]}")
            self.debug_logger.debug(f"  Selected hoop frame: {self.selected_hoop[1]}")
            self.debug_logger.debug(f"  Ball: {self.selected_ball[0]} conf={self.selected_ball[4]:.2f}")
            self.debug_logger.debug(f"  Hoop: {self.selected_hoop[0]} conf={self.selected_hoop[4]:.2f}")

            # Add to trajectory arrays (now guaranteed to be synchronized)
            self.ball_tracker.add_ball_position(selected_ball_data, self.frame_count)
            self.hoop_tracker.add_hoop_position(selected_hoop_data, self.frame_count)

            # Perform UP/DOWN detection with synchronized data
            self.shot_detection_with_selected()

        else:
            # No synchronized detection available in current frame
            self.selected_ball = None
            self.selected_hoop = None
            
            # Check for net disturbance based shot detection
            # This will follow the exact sequence:
            # 1. First check if no valid ball detected in current frame (no selected ball) - already confirmed
            # 2. Then check if there was an UP state before
            # 3. Then check if the hoop is at the edge of the video
            # 4. Then predict ball position using trajectory fitting
            # 5. If predicted position is in hoop, compare hoop region between frames for net disturbance
            # 6. If disturbance is detected, count as successful shot
            if self.check_hoop_disturbance_shot():
                # Record the shot based on net disturbance
                self.shot_analyzer.down = True
                self.shot_analyzer.down_frame = self.frame_count
                self.debug_logger.info(f"🏀 基于篮网扰动检测到进球，帧号: {self.frame_count}")
                self.analyze_shot_attempt()
            # Do not perform UP/DOWN detection, do not add to trajectory
            
    def analyze_shot_attempt(self):
        """Analyze shot attempt with safe state handling"""
        result = self.shot_analyzer.analyze_shot_attempt(
            self.ball_tracker.ball_pos, 
            self.hoop_tracker.hoop_pos
        )
        
        # Update our local counters
        self.makes = self.shot_analyzer.makes
        self.attempts = self.shot_analyzer.attempts
        self.overlay_color = self.shot_analyzer.overlay_color
        self.overlay_text = self.shot_analyzer.overlay_text
        self.fade_counter = self.shot_analyzer.fade_counter
        
        timestamp = self.frame_count / 30
        
        # 安全获取球和篮筐位置
        ball_pos = None
        if self.ball_tracker.ball_pos:
            ball_pos = self.ball_tracker.ball_pos[-1][0]
        
        hoop_pos = None
        hoop_confidence = 0.0
        if self.hoop_tracker.hoop_pos:
            hoop_pos = self.hoop_tracker.hoop_pos[-1][0]
            hoop_confidence = self.hoop_tracker.hoop_pos[-1][4]
        
        self.shot_logger.log_shot(
            frame_idx=self.frame_count,
            timestamp=timestamp,
            ball_pos=ball_pos,
            hoop_pos=hoop_pos,
            ball_confidence=self.ball_tracker.ball_pos[-1][4] if self.ball_tracker.ball_pos else 0.0,
            is_successful=result['is_successful'],
            debug_info=result['debug_info']
        )

        # 清理轨迹数据
        self.ball_tracker.ball_pos = []
        self.hoop_tracker.hoop_pos = []
        if self.enable_person_detection:
            self.person_pos = []
            
    def shot_detection_with_selected(self):
        """
        Perform UP/DOWN detection using synchronized selected data
        Only called when both ball and hoop are detected in same frame
        """
        if not self.selected_ball or not self.selected_hoop:
            return

        # 🔧 修复：强制保留UP状态，防止被后续帧误清
        # 使用局部变量记录UP状态，避免全局被重置
        has_cached_up = self.shot_analyzer.up
        has_cached_up_frame = self.shot_analyzer.up_frame

        # UP detection（只有在未缓存状态下检测）
        if not has_cached_up:
            up_detected = self.shot_analyzer.detect_up(self.selected_ball, self.selected_hoop)
            if up_detected:
                self.shot_analyzer.up = True
                self.shot_analyzer.up_frame = self.frame_count
                has_cached_up = True
                has_cached_up_frame = self.frame_count
                # 记录UP事件
                self.frame_logger.log_up_event(self.frame_count)

        # DOWN detection（只有在确认UP后才检测）
        if has_cached_up and not self.shot_analyzer.down:
            down_detected = self.shot_analyzer.detect_down(self.selected_ball, self.selected_hoop)
            if down_detected:
                self.shot_analyzer.down = True
                self.shot_analyzer.down_frame = self.frame_count
                # 记录DOWN事件
                self.frame_logger.log_down_event(self.frame_count)
                # 🔧 修复：在进行投篮分析前，确保将当前帧的球添加到轨迹中
                # 按照轨迹分析数据完整性规范，DOWN帧检测到的球必须被添加到轨迹数据中
                if self.selected_ball and self.selected_hoop:
                    # 确保当前选中的球和篮筐数据被添加到轨迹数组
                    self.ball_tracker.ball_pos.append(self.selected_ball)
                    self.hoop_tracker.hoop_pos.append(self.selected_hoop)
                
                # ✅ 立即触发投篮分析，避免状态丢失
                self.analyze_shot_attempt()

    def display_score(self):
        # Add text
        text = str(self.makes) + " / " + str(self.attempts)
        cv2.putText(self.frame, text, (50, 125), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 6)
        cv2.putText(self.frame, text, (50, 125), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 3)

        # Display frame number in the top-left corner
        frame_text = f"Frame: {self.frame_count}"
        cv2.putText(self.frame, frame_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 4)
        cv2.putText(self.frame, frame_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 2)

        # Draw UP/DOWN regions for visualization
        self.draw_detection_regions()

        # Add overlay text for shot result if it exists
        if hasattr(self, 'overlay_text'):
            # Calculate text size to position it at the right top corner
            (text_width, text_height), _ = cv2.getTextSize(self.overlay_text, cv2.FONT_HERSHEY_SIMPLEX, 3, 6)
            text_x = self.frame.shape[1] - text_width - 40  # Right alignment with some margin
            text_y = 100  # Top margin

            # Display overlay text with color (overlay_color)
            cv2.putText(self.frame, self.overlay_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 3,
                        self.overlay_color, 6)

        # Gradually fade out color after shot
        if self.fade_counter > 0:
            alpha = 0.2 * (self.fade_counter / self.fade_frames)
            self.frame = cv2.addWeighted(self.frame, 1 - alpha, np.full_like(self.frame, self.overlay_color), alpha, 0)
            self.fade_counter -= 1

    def draw_detection_regions(self):
        """
        Draw visualization regions for UP/DOWN detection
        """
        # This is a placeholder - implement actual visualization if needed
        pass

    def run(self):
        # Initialize video writer if output path is provided
        if self.output_video:
            ret, frame = self.cap.read()
            if ret:
                height, width = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                self.video_writer = cv2.VideoWriter(self.output_video, fourcc, 30.0, (width, height))
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Rewind to first frame
        
        # 缓存帧尺寸，避免重复计算
        self.frame_width = 0
        self.frame_height = 0

        # Initialize progress bar
        progress_bar = tqdm(total=self.total_frames, desc="Processing Video", unit='frames')

        while True:
            ret, self.frame = self.cap.read()

            if not ret:
                # End of the video or an error occurred
                if self.video_writer:
                    self.video_writer.release()
                break

            # 更新帧尺寸缓存
            if self.frame is not None:
                self.frame_height, self.frame_width = self.frame.shape[:2]


            # Run detection models
            if self.use_shared_model:
                # Use shared model for both ball and person detection
                main_results = self.ball_model(self.frame, stream=True, device=self.device, verbose=False)
                hoop_results = self.hoop_model(self.frame, stream=True, device=self.device, verbose=False)
            else:
                # Use separate models
                main_results = self.ball_model(self.frame, stream=True, device=self.device, verbose=False)
                hoop_results = self.hoop_model(self.frame, stream=True, device=self.device, verbose=False)
                person_results = self.person_model(self.frame, stream=True, device=self.device, verbose=False)



            # Collect all detections in current frame for logging
            current_frame_balls = []
            current_frame_hoops = []
            current_frame_persons = []

            # Process main model detections (ball and person if shared model)
            for r in main_results:
                boxes = r.boxes
                if boxes is not None:
                    for box in boxes:
                        # Bounding box
                        x1, y1, x2, y2 = box.xyxy[0]
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        w, h = x2 - x1, y2 - y1

                        # Confidence
                        conf = math.ceil((box.conf[0] * 100)) / 100

                        # Class Name from ball model
                        cls = int(box.cls[0])
                        if cls < len(self.ball_model_classes):
                            current_class = self.ball_model_classes[cls]
                        else:
                            current_class = f"Unknown_{cls}"

                        center = (int(x1 + w / 2), int(y1 + h / 2))

                        # Check if this is a sports ball (basketball) using unified mapping
                        is_ball = object_class_mapper.is_object_type(current_class, 'ball')

                        # Check if this is a person (when using shared model)
                        is_person = object_class_mapper.is_object_type(current_class, 'person')

                        if is_ball:
                            # Calculate ball area for size filtering
                            ball_area = w * h

                            current_frame_balls.append({
                                "bbox": [x1, y1, x2, y2],
                                "center": center,
                                "size": {"width": w, "height": h},
                                "confidence": float(conf),
                                "class": current_class,
                                "area": ball_area
                            })

                            # Draw detection rectangle for valid balls
                            if ball_area >= self.min_ball_area and (conf > 0.4 or (in_hoop_region(center, self.hoop_tracker.hoop_pos) and conf > 0.2)):
                                # Green box for all confidence detected balls
                                cvzone.cornerRect(self.frame, (x1, y1, w, h), colorC=(0, 255, 0), t=2)
                                self.debug_logger.debug_file_only(f"绘制普通球边界框: center={center}, frame={self.frame_count}")
                            elif ball_area < self.min_ball_area:
                                # Draw filtered out balls in gray for debugging
                                cvzone.cornerRect(self.frame, (x1, y1, w, h), colorC=(128, 128, 128), t=1)
                                cvzone.putTextRect(self.frame, f'Small Ball {ball_area}px', (x1, y1-10),
                                                 scale=0.6, thickness=1, colorR=(128, 128, 128))

                        elif is_person and self.enable_person_detection and self.use_shared_model and conf > 0.3:
                            # Process person detection from shared model (only if enabled)
                            current_frame_persons.append({
                                "bbox": [x1, y1, x2, y2],
                                "center": center,
                                "size": {"width": w, "height": h},
                                "confidence": float(conf),
                                "class": current_class
                            })

            # Process hoop detections from custom model
            for r in hoop_results:

                boxes = r.boxes
                if boxes is not None:
                    for box in boxes:
                        # Bounding box
                        x1, y1, x2, y2 = box.xyxy[0]
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        w, h = x2 - x1, y2 - y1

                        # Confidence
                        conf = math.ceil((box.conf[0] * 100)) / 100

                        # Class Name from hoop model
                        cls = int(box.cls[0])
                        if cls < len(self.hoop_model_classes):
                            current_class = self.hoop_model_classes[cls]
                        else:
                            current_class = f"Unknown_{cls}"

                        center = (int(x1 + w / 2), int(y1 + h / 2))

                        # Check if this is a basketball hoop or rim using unified mapping
                        is_hoop = object_class_mapper.is_object_type(current_class, 'hoop')

                        if is_hoop:
                            current_frame_hoops.append({
                                "bbox": [x1, y1, x2, y2],
                                "center": center,
                                "size": {"width": w, "height": h},
                                "confidence": float(conf),
                                "class": current_class
                            })

                            # Draw detection rectangle for valid hoops (will be added to trajectory via process_frame_detections)
                            if conf > 0.4:
                                cvzone.cornerRect(self.frame, (x1, y1, w, h), colorC=(0, 255, 255), t=3)

            # Process person detections with overlap filtering
            raw_person_detections = []

            # If using separate person model, process its results
            if not self.use_shared_model and self.enable_person_detection and self.person_model:
                person_boxes = None
                for r in person_results:
                    person_boxes = r.boxes
                    if person_boxes is not None:
                        for box in person_boxes:
                            # Bounding box
                            x1, y1, x2, y2 = box.xyxy[0]
                            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                            w, h = x2 - x1, y2 - y1

                            # Confidence
                            conf = math.ceil((box.conf[0] * 100)) / 100

                            # Class Name from person model
                            cls = int(box.cls[0])
                            if cls < len(self.person_model_classes):
                                current_class = self.person_model_classes[cls]
                            else:
                                current_class = f"Unknown_{cls}"

                            center = (int(x1 + w / 2), int(y1 + h / 2))

                            # Check if this is a person using unified mapping
                            is_person = object_class_mapper.is_object_type(current_class, 'person')

                            if is_person and conf > 0.3:
                                # Only add to raw_person_detections if using separate model and person detection is enabled
                                raw_person_detections.append({
                                    "bbox": [x1, y1, x2, y2],
                                    "center": center,
                                    "size": {"width": w, "height": h},
                                    "confidence": float(conf),
                                    "class": current_class
                                })

            # Process person detections only if enabled
            if self.enable_person_detection:
                # If using shared model, person detections are already in current_frame_persons
                # If using separate model, add the raw detections to the list
                if not self.use_shared_model:
                    # Filter overlapping detections - prefer full body (larger height)
                    filtered_person_detections = self.filter_overlapping_persons(raw_person_detections)
                    current_frame_persons.extend(filtered_person_detections)
                else:
                    # For shared model, filter the detections already collected
                    filtered_person_detections = self.filter_overlapping_persons(current_frame_persons)
                    current_frame_persons = filtered_person_detections

                # Add filtered detections to trajectory and draw them
                for person_detection in current_frame_persons:

                    # Add to trajectory
                    center = person_detection["center"]
                    w = person_detection["size"]["width"]
                    h = person_detection["size"]["height"]
                    conf = person_detection["confidence"]

                    self.person_pos.append((center, self.frame_count, w, h, conf))

                    # Draw bounding box and label
                    x1, y1, x2, y2 = person_detection["bbox"]
                    cvzone.cornerRect(self.frame, (x1, y1, x2-x1, y2-y1), colorC=(0, 255, 0), t=2)
                    cvzone.putTextRect(self.frame, f'Person {conf:.2f}', (x1, y1-10),
                                     scale=0.8, thickness=1, colorR=(0, 255, 0))

            # Update prev_frame for net disturbance detection
            # 移除这里的更新，因为已经在 process_frame_detections 中更新了

            # Check for scene change using frame difference (additional method)
            if self.prev_frame is not None and self.frame is not None:
                # 检查当前帧是否是预先检测到的场景变化帧（排除第0帧）
                if self.frame_count in self.scene_change_frames and self.frame_count > 0:
                    self.debug_logger.warning(f"⚠️ Frame {self.frame_count}: 检测到场景变化，可能是视频剪辑，重置跟踪数据")
                    # 记录视频剪辑事件（帧索引从0开始）
                    self.frame_logger.log_video_cut(self.frame_count)
                    self.ball_tracker.reset_tracking()
                    self.hoop_tracker.reset_tracking()
                    self.person_pos = []
                    self.shot_analyzer.reset_shot_detection()
                    # Continue processing without returning, as we still want to process current frame detections
            # First clean existing motion data
            self.clean_motion()
            
            # Then process frame detections and perform synchronized UP/DOWN analysis
            self.debug_logger.debug(f"📍 Processing frame {self.frame_count} with {len(current_frame_balls)} balls, {len(current_frame_hoops)} hoops")
            self.process_frame_detections(current_frame_balls, current_frame_hoops)

            if self.selected_ball:
                x1, y1 = self.selected_ball[0]
                w = self.selected_ball[2]
                h = self.selected_ball[3]
                x1 = int(x1-w/2)
                y1 = int(y1-h/2)
                w = int(w)
                h = int(h)
                cvzone.cornerRect(self.frame, (x1, y1, w, h), colorC=(255, 0, 0), t=3)
            # Draw ball trajectory points (red dots)
            for pos in self.ball_tracker.ball_pos:
                center = pos[0]
                cv2.circle(self.frame, center, 3, (0, 0, 255), -1)  # Red dots for history positions
            
            # Draw predicted ball position if available (orange dot)
            predicted_position = self.predict_ball_position_from_trajectory(self.frame_count)
            if predicted_position:
                pred_x, pred_y = int(predicted_position[0]), int(predicted_position[1])
                cv2.circle(self.frame, (pred_x, pred_y), 5, (0, 165, 255), -1)  # Orange dot for predicted position
            
            self.display_score()
            self.frame_logger.frame_count = self.frame_count
            self.frame_logger.update_progress(self.frame_count, self.total_frames)
            progress_bar.update(1)

            # Update frame logger with current frame count
            self.frame_logger.frame_count = self.frame_count

            # Log frame data after processing
            all_balls = self.ball_tracker.ball_pos if hasattr(self.ball_tracker, 'ball_pos') else []
            all_hoops = self.hoop_tracker.hoop_pos if hasattr(self.hoop_tracker, 'hoop_pos') else []
            all_persons = self.person_pos if hasattr(self, 'person_pos') and self.enable_person_detection else []

            # Determine selected indices based on new synchronized detection logic
            # selected_ball and selected_hoop now represent the best detections from current frame
            selected_ball_idx = len(all_balls) - 1 if all_balls and self.selected_ball else -1
            selected_hoop_idx = len(all_hoops) - 1 if all_hoops and self.selected_hoop else -1
            selected_person_idx = len(all_persons) - 1 if all_persons else -1

            # Only pass person data if person detection is enabled
            persons_data = all_persons if self.enable_person_detection else []
            current_persons_data = current_frame_persons if self.enable_person_detection else []

            self.frame_logger.log_frame_data(
                self.frame_count,
                all_balls,
                all_hoops,
                persons_data,
                selected_ball_idx,
                selected_hoop_idx,
                selected_person_idx,
                current_frame_balls,
                current_frame_hoops,
                current_persons_data,
                self.selected_ball,
                self.selected_hoop
            )

            # Write frame to output video if specified
            if self.video_writer:
                self.video_writer.write(self.frame)
            else:
                cv2.imshow('Frame', self.frame)
                # Close if 'q' is clicked
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            self.frame_count += 1

        progress_bar.close()
        self.cap.release()
        if self.video_writer:
            self.video_writer.release()
        else:
            cv2.destroyAllWindows()

        # Close frame log file if it exists and is still open
        if hasattr(self.frame_logger, '_frame_log_file') and self.frame_logger._frame_log_file and not self.frame_logger._frame_log_file.closed:
            try:
                self.frame_logger._frame_log_file.write(']')  # Close JSON array
                self.frame_logger._frame_log_file.close()
            except (ValueError, AttributeError) as e:
                print(f"Warning: Could not close frame log file properly: {e}")

        # 处理可能未完成的UP事件
        self.frame_logger.finalize_incomplete_up()

        # Save both frame and shot logs after processing completes
        frame_log_filename = self.frame_logger.save_log()
        shot_log_filename = self.shot_logger.save_log()
        print(f"\n✅ 帧日志已保存到: {frame_log_filename}")
        print(f"\n✅ 投篮日志已保存到: {shot_log_filename}")

        # 打印改进的摘要
        self.shot_logger.print_improved_summary()

        # 关闭调试日志器
        self.debug_logger.info(f"Processing completed. Debug log saved to: {self.debug_logger.debug_log_file}")
        self.debug_logger.close()
        print(f"\n📝 调试日志已保存到: {self.debug_logger.debug_log_file}")

if __name__ == "__main__":
    import argparse
    from model_configs import get_model_config, list_all_configs

    parser = argparse.ArgumentParser(description='Basketball Shot Detector with Enhanced Model Support')
    parser.add_argument('--input', type=str, default='video_test_5.mp4', help='Input video file path')
    parser.add_argument('--output', type=str, help='Output video file path')
    parser.add_argument('--ball-model', type=str, default='Yolo-Weights/yolov8x.pt', help='Ball detection model (default: yolov8x.pt)')
    parser.add_argument('--hoop-model', type=str, default= None, help='Hoop detection model (default: best.pt)')
    parser.add_argument('--config', type=str, help='Use predefined model configuration (e.g., high_accuracy, balanced, real_time)')
    parser.add_argument('--list-models', action='store_true', help='List all available model configurations')
    args = parser.parse_args()

    # List models if requested
    if args.list_models:
        list_all_configs()
        exit(0)

    # Use config if specified
    ball_model = args.ball_model
    if args.config:
        config = get_model_config(args.config)
        if config:
            ball_model = config['ball_model']
            print(f"🎯 Using configuration '{args.config}': {config['description']}")
            print(f"📋 Ball model: {ball_model}")
        else:
            print("❌ Invalid configuration. Use --list-models to see available options.")
            exit(1)

    # Create detector with dual models
    detector = ShotDetector(args.input, args.output, ball_model, args.hoop_model)
    detector.run() 