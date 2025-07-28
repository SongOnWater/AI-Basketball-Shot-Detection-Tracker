# Avi Shah - Basketball Shot Detector/Tracker - July 2023

from ultralytics import YOLO
import cv2
import cvzone
import math
import numpy as np
import json
import time
from tqdm import tqdm
import math
from utils import score, in_hoop_region, clean_hoop_pos, clean_ball_pos, get_device
from datetime import datetime
import logging
import os

from logging_utils import DebugLogger, ShotLogger


class ShotDetector:
    def __init__(self, input_video="video_test_5.mp4", output_video=None, ball_model_path="yolov8m.pt", hoop_model_path=None, person_model_path=None, use_shared_model=True, min_ball_area=400, enable_person_detection=False, model_config=None):
        import os
        # Load models for optimal detection
        self.overlay_text = "Waiting..."
        self.use_shared_model = use_shared_model

        # 不再在 ShotDetector 内部创建 debug_logger
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

        # Person detection: only load if enabled
        if enable_person_detection:
            if use_shared_model:
                self.person_model = self.ball_model  # Share the same model
                self.person_model_path = ball_model_path
                print(f"👤 Using shared model for person detection: {ball_model_path}")
            else:
                # Use separate person model if specified
                self.person_model_path = person_model_path or "yolov8n.pt"
                self.person_model = YOLO(self.person_model_path)
                print(f"👤 Loaded separate person model: {self.person_model_path}")
        else:
            self.person_model = None
            self.person_model_path = None
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
            output_video = f"{video_name}_{model_name}_output_{timestamp}.mp4"
        self.output_video = output_video
        self.video_writer = None

        self.logger = ShotLogger(input_file=self.input_video, model_path=self.ball_model_path, log_type="frame")
        # 关键：让 ShotDetector 直接引用 ShotLogger 的 debug_logger
        self.debug_logger = DebugLogger(debug_log_file=os.path.join('logs', 'console_output.log'), input_file=self.input_video, model_path=self.ball_model_path, log_type="debug")
        self.debug_logger.debug("[ShotDetector] debug_logger now references ShotLogger's initialized logger.")
        # ...existing code...
        # Uncomment this line to accelerate inference. Note that this may cause errors in some setups.
        #self.model.half()
        
        # Initialize class names for both models
        self.class_names = ['Basketball', 'Basketball Hoop', 'Rim']  # Extended with Rim detection

        # Get class names from both models
        if hasattr(self.ball_model, 'names'):
            self.ball_model_classes = self.ball_model.names
            print(f"📋 Ball model classes: {len(self.ball_model_classes)} classes (including 'sports ball')")
        else:
            self.ball_model_classes = {0: "Basketball"}

        if hasattr(self.hoop_model, 'names'):
            self.hoop_model_classes = self.hoop_model.names
            # Filter and show only hoop-related classes that will be used for detection
            hoop_classes = [cls for cls in self.hoop_model_classes.values()
                           if 'hoop' in cls.lower() or 'rim' in cls.lower() or cls.lower() == 'basketball hoop']
            
            # 显示所有支持的篮筐类别，包括代码中硬编码的类别
            supported_hoop_classes = ["Basketball Hoop", "hoop", "Rim"] + hoop_classes
            # 去重
            supported_hoop_classes = list(set(supported_hoop_classes))
            
            print(f"📋 Hoop model classes: {list(self.hoop_model_classes.values())}")
            print(f"🎯 Active hoop detection classes: {supported_hoop_classes}")
        else:
            self.hoop_model_classes = {0: "Basketball", 1: "Basketball Hoop"}

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

        self.ball_pos = []  # array of tuples ((x_pos, y_pos), frame count, width, height, conf)
        self.hoop_pos = []  # array of tuples ((x_pos, y_pos), frame count, width, height, conf)
        self.person_pos = []  # array of tuples ((x_pos, y_pos), frame count, width, height, conf)

        # Selected detections for UP/DOWN analysis (synchronized same-frame data)
        self.selected_ball = None  # Best ball detection from current frame for UP/DOWN analysis
        self.selected_hoop = None  # Best hoop detection from current frame for UP/DOWN analysis

        self.frame_count = 1
        self.frame = None

        self.makes = 0
        self.attempts = 0

        # Used to detect shots (upper and lower region)
        self.up = False
        self.down = False
        self.up_frame = 0
        self.down_frame = 0

        # Used for green and red colors after make/miss
        self.fade_frames = 20
        self.fade_counter = 0
        self.overlay_color = (0, 0, 0)

        # Ball filtering parameters
        self.min_ball_area = min_ball_area  # Minimum ball area in pixels (width * height)


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
            if hasattr(self, 'frame') and self.frame is not None:
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
        # 需要至少3个历史点来进行有效的轨迹拟合
        if len(self.ball_pos) < 3:
            self.debug_logger.debug(f"历史轨迹点不足，无法进行拟合预测: {len(self.ball_pos)} 点")
            return None
            
        # 获取最近的N个历史点（最多10个点）
        recent_history = self.ball_pos[-10:]
        
        # 提取坐标和帧号
        x_coords = [pos[0][0] for pos in recent_history]
        y_coords = [pos[0][1] for pos in recent_history]
        frames = [pos[1] for pos in recent_history]
        
        # 检查是否有足够的不同帧
        if len(set(frames)) < 3:
            self.debug_logger.debug(f"历史轨迹中不同帧数不足，无法进行拟合预测")
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
            
            self.debug_logger.debug(f"轨迹拟合预测位置: ({predicted_x:.1f}, {predicted_y:.1f}) 在帧 {current_frame}")
            
            return (predicted_x, predicted_y)
        except Exception as e:
            self.debug_logger.warning(f"轨迹拟合预测失败: {e}")
            return None

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
            
        # 尝试使用轨迹拟合进行筛选（对每个球都进行检验）
        predicted_position = self.predict_ball_position_from_trajectory(self.frame_count)
        
        if predicted_position:
            # 设置基础偏差阈值（像素）
            base_deviation_threshold = 50  # 基础阈值
            
            # 获取拟合数据的帧序列最大值
            recent_history = self.ball_pos[-10:]
            frames = [pos[1] for pos in recent_history]
            max_history_frame = max(frames) if frames else self.frame_count
            
            # 计算当前帧与拟合数据帧序列最大值的差值，并动态调整阈值
            frame_diff = abs(self.frame_count - max_history_frame)
            # 每帧差异增加基础阈值的比例
            deviation_threshold = base_deviation_threshold * (1 + frame_diff)
            
            self.debug_logger.debug(f"动态偏差阈值: {deviation_threshold:.1f}px (基础阈值: {base_deviation_threshold}px, 帧差值: {frame_diff})")
            
            # 计算每个球与预测位置的偏差
            for ball in quality_balls:
                ball_center = ball['center']
                deviation = ((ball_center[0] - predicted_position[0])**2 + 
                            (ball_center[1] - predicted_position[1])**2)**0.5
                ball['trajectory_deviation'] = deviation
                
                self.debug_logger.debug(f"球 ({ball_center[0]:.1f}, {ball_center[1]:.1f}) 与预测位置偏差: {deviation:.1f}px")
            
            # 过滤掉偏差超过阈值的球
            trajectory_filtered_balls = [ball for ball in quality_balls 
                                       if ball.get('trajectory_deviation', float('inf')) <= deviation_threshold]
            
            if trajectory_filtered_balls:
                self.debug_logger.debug(f"轨迹筛选后剩余 {len(trajectory_filtered_balls)}/{len(quality_balls)} 个球")
                # 从轨迹筛选后的球中选择置信度最高的
                best_ball = max(trajectory_filtered_balls, key=lambda x: x['confidence'])
                self.debug_logger.debug(f"选择基于轨迹筛选的最佳球: 置信度={best_ball['confidence']:.2f}, 偏差={best_ball.get('trajectory_deviation', 'N/A'):.1f}px")
            else:
                # 如果所有球都被轨迹筛选过滤掉，则丢弃所有球
                self.debug_logger.debug(f"所有球都超出轨迹偏差阈值，丢弃所有球")
                self.debug_logger.debug(f"最大偏差: {max([ball.get('trajectory_deviation', float('inf')) for ball in quality_balls]):.1f}px, 动态阈值: {deviation_threshold:.1f}px (基础阈值: {base_deviation_threshold}px, 帧差值: {frame_diff})")
                best_ball = None
        else:
            # 无法进行轨迹预测时，使用置信度最高的球
            self.debug_logger.debug(f"无法进行轨迹预测，使用置信度最高的球")
            best_ball = max(quality_balls, key=lambda x: x['confidence'])

        # 选择置信度最高的篮筐
        best_hoop = max(quality_hoops, key=lambda x: x['confidence'])

        return best_ball, best_hoop

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

        self.debug_logger.debug(f"🔥 FORCE DEBUG: process_frame_detections called for frame {self.frame_count}")
        self.debug_logger.debug(f"🔥 Input: {len(current_frame_balls)} balls, {len(current_frame_hoops)} hoops")

        # Check for significant hoop position/size changes (possible video cut)
        if len(self.hoop_pos) > 1 and len(current_frame_hoops) > 0:
            last_hoop = self.hoop_pos[-1]
            current_hoop = max(current_frame_hoops, key=lambda x: x['confidence'])
            
            # Calculate position and size differences
            pos_diff = math.sqrt((last_hoop[0][0] - current_hoop['center'][0])**2 + 
                                (last_hoop[0][1] - current_hoop['center'][1])**2)
            size_diff = abs(last_hoop[2] - current_hoop['size']['width']) + \
                       abs(last_hoop[3] - current_hoop['size']['height'])
            
            # Thresholds for significant change (adjust as needed)
            pos_threshold = 0.5 * math.sqrt(last_hoop[2]**2 + last_hoop[3]**2)
            size_threshold = 0.5 * (last_hoop[2] + last_hoop[3])
            
            if pos_diff > pos_threshold or size_diff > size_threshold:
                self.debug_logger.warning(f"⚠️ 检测到篮筐位置/大小显著变化 (位置差: {pos_diff:.1f}px > {pos_threshold:.1f}px 或大小差: {size_diff:.1f}px > {size_threshold:.1f}px)，可能是视频剪辑，重置跟踪数据")
                self.ball_pos = []
                self.hoop_pos = []
                self.person_pos = []
                self.up = False
                self.down = False
                self.up_frame = 0
                self.down_frame = 0
                self.selected_ball = None
                self.selected_hoop = None
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
            self.ball_pos.append(self.selected_ball)
            self.hoop_pos.append(self.selected_hoop)

            # Perform UP/DOWN detection with synchronized data
            self.shot_detection_with_selected()

        else:
            # No synchronized detection available
            self.selected_ball = None
            self.selected_hoop = None
            # Do not perform UP/DOWN detection, do not add to trajectory

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

    def run(self):
        # Initialize video writer if output path is provided
        if self.output_video:
            ret, frame = self.cap.read()
            if ret:
                height, width = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                self.video_writer = cv2.VideoWriter(self.output_video, fourcc, 30.0, (width, height))
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Rewind to first frame

        # Initialize progress bar
        progress_bar = tqdm(total=self.total_frames, desc="Processing Video", unit='frames')

        while True:
            ret, self.frame = self.cap.read()

            if not ret:
                # End of the video or an error occurred
                if self.video_writer:
                    self.video_writer.release()
                break


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



            # ...existing code...

            # Collect all detections in current frame for logging
            current_frame_balls = []
            current_frame_hoops = []
            current_frame_persons = []

            # ...existing code for detection collection...


            # ...existing code for detection collection...


            # === 每帧详细debug输出（必须在所有检测append和selected_ball/hoop赋值后） ===
            # 先处理main/hoop/person detection append...
            # 再选出selected_ball/selected_hoop（在process_frame_detections前赋值）
            # 这里每帧都写入debug log
            self.debug_logger.debug(f"Frame {self.frame_count}: balls={len(current_frame_balls)}, hoops={len(current_frame_hoops)}, persons={len(current_frame_persons)}")
            if hasattr(self, 'selected_ball') and self.selected_ball:
                self.debug_logger.debug(f"  Selected ball: pos={self.selected_ball[0]}, conf={self.selected_ball[4]:.2f}")
            else:
                self.debug_logger.debug(f"  Selected ball: None")
            if hasattr(self, 'selected_hoop') and self.selected_hoop:
                self.debug_logger.debug(f"  Selected hoop: pos={self.selected_hoop[0]}, conf={self.selected_hoop[4]:.2f}")
            else:
                self.debug_logger.debug(f"  Selected hoop: None")

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

                        # Check if this is a sports ball (basketball)
                        is_ball = (current_class in ["Basketball", "sports ball"] or
                                  "ball" in current_class.lower())

                        # Check if this is a person (when using shared model)
                        is_person = (current_class.lower() == "person")

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

                            # Draw detection rectangle for valid balls (will be added to trajectory via process_frame_detections)
                            if ball_area >= self.min_ball_area and (conf > 0.2 or (in_hoop_region(center, self.hoop_pos) and conf > 0.1)):
                                cvzone.cornerRect(self.frame, (x1, y1, w, h), colorC=(255, 0, 0), t=3)
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

                        # Check if this is a basketball hoop or rim
                        is_hoop = (
                            current_class in ["Basketball Hoop", "hoop", "Rim"] or
                            "hoop" in current_class.lower() or
                            "rim" in current_class.lower() or
                            (current_class.lower() == "basketball hoop")
                        )

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
            if not self.use_shared_model:
                for r in person_results:
                    boxes = r.boxes
                if boxes is not None:
                    for box in boxes:
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

                        # Check if this is a person
                        is_person = (current_class.lower() == "person")

                        if is_person and self.enable_person_detection and not self.use_shared_model and conf > 0.3:
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

            # First clean existing motion data
            self.clean_motion()

            # Then process frame detections and perform synchronized UP/DOWN analysis
            self.debug_logger.debug(f"📍 Processing frame {self.frame_count} with {len(current_frame_balls)} balls, {len(current_frame_hoops)} hoops")
            self.process_frame_detections(current_frame_balls, current_frame_hoops)
            self.display_score()
            self.logger.frame_count = self.frame_count
            self.logger.update_progress(self.frame_count, self.total_frames)
            progress_bar.update(1)

            # Write frame to output video if specified
            if self.video_writer:
                self.video_writer.write(self.frame)
            else:
                cv2.imshow('Frame', self.frame)
                # Close if 'q' is clicked
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # Log frame data after processing
            all_balls = self.ball_pos if hasattr(self, 'ball_pos') else []
            all_hoops = self.hoop_pos if hasattr(self, 'hoop_pos') else []
            all_persons = self.person_pos if hasattr(self, 'person_pos') and self.enable_person_detection else []

            # Determine selected indices based on new synchronized detection logic
            # selected_ball and selected_hoop now represent the best detections from current frame
            selected_ball_idx = len(all_balls) - 1 if all_balls and self.selected_ball else -1
            selected_hoop_idx = len(all_hoops) - 1 if all_hoops and self.selected_hoop else -1
            selected_person_idx = len(all_persons) - 1 if all_persons else -1

            # Only pass person data if person detection is enabled
            persons_data = all_persons if self.enable_person_detection else []
            current_persons_data = current_frame_persons if self.enable_person_detection else []

            self.logger.log_frame_data(
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
            self.frame_count += 1

        progress_bar.close()
        self.cap.release()
        if self.video_writer:
            self.video_writer.release()
        else:
            cv2.destroyAllWindows()

        # Close frame log file if it exists and is still open
        if hasattr(self.logger, '_frame_log_file') and self.logger._frame_log_file and not self.logger._frame_log_file.closed:
            try:
                self.logger._frame_log_file.write(']')  # Close JSON array
                self.logger._frame_log_file.close()
            except (ValueError, AttributeError) as e:
                print(f"Warning: Could not close frame log file properly: {e}")

        # Save shot log after processing completes
        log_filename = self.logger.save_log()
        print(f"\n✅ 投篮日志已保存到: {log_filename}")

        # 打印改进的摘要
        self.logger.print_improved_summary()

        # 关闭调试日志器
        self.debug_logger.info(f"Processing completed. Debug log saved to: {self.debug_logger.debug_log_file}")
        self.debug_logger.close()
        print(f"\n📝 调试日志已保存到: {self.debug_logger.debug_log_file}")

    def clean_motion(self):
        # Clean and display ball motion
        self.ball_pos = clean_ball_pos(self.ball_pos, self.frame_count)
        for i in range(0, len(self.ball_pos)):
            cv2.circle(self.frame, self.ball_pos[i][0], 2, (0, 0, 255), 2)

        # Clean hoop motion and display current hoop center
        if len(self.hoop_pos) > 1:
            self.hoop_pos = clean_hoop_pos(self.hoop_pos)

        # 🔧 SYNC FIX: Draw hoop center using synchronized drawing logic
        should_draw, hoop_data, data_source = self.should_draw_hoop_and_regions()
        if should_draw and hoop_data:
            hoop_center = hoop_data[0]
            cv2.circle(self.frame, hoop_center, 2, (128, 128, 0), 2)
            self.debug_logger.debug(f"🎨 Drawing hoop center using {data_source}")

    def detect_up_with_selected(self, selected_ball, selected_hoop):
        """
        Detect UP state using synchronized selected data from same frame

        Args:
            selected_ball: Ball detection tuple from current frame
            selected_hoop: Hoop detection tuple from current frame

        Returns:
            bool: True if ball is in UP region relative to hoop
        """
        if not selected_ball or not selected_hoop:
            self.debug_logger.debug(f"🚫 UP detection skipped - missing data: ball={bool(selected_ball)}, hoop={bool(selected_hoop)}")
            return False

        # Ensure same frame (double safety check)
        if selected_ball[1] != selected_hoop[1]:
            self.debug_logger.warning(f"⚠️ Frame mismatch in UP detection - ball:{selected_ball[1]}, hoop:{selected_hoop[1]}")
            return False

        # Extract positions and dimensions
        ball_x, ball_y = selected_ball[0]
        hoop_x, hoop_y = selected_hoop[0]
        hoop_w, hoop_h = selected_hoop[2], selected_hoop[3]

        # Calculate UP region boundaries - STRICTER DEFINITION
        # Based on observation that current definition is too loose
        x1 = hoop_x - 3 * hoop_w      # Reduced from 4x to 3x width
        x2 = hoop_x + 3 * hoop_w      # Reduced from 4x to 3x width
        y1 = hoop_y - 1.5 * hoop_h    # Reduced from 2x to 1.5x height
        y2 = hoop_y - 0.8 * hoop_h    # Increased from 0.5x to 0.8x height (smaller region)

        is_in_up_region = x1 < ball_x < x2 and y1 < ball_y < y2

        self.debug_logger.debug(f"🔍 UP detection Frame {selected_ball[1]}:")
        self.debug_logger.debug(f"  Ball: ({ball_x}, {ball_y})")
        self.debug_logger.debug(f"  Hoop: ({hoop_x}, {hoop_y}) {hoop_w}×{hoop_h}")
        self.debug_logger.debug(f"  UP region: X({x1:.0f}-{x2:.0f}) Y({y1:.0f}-{y2:.0f})")
        self.debug_logger.debug(f"  X check: {x1:.0f} < {ball_x} < {x2:.0f} = {x1 < ball_x < x2}")
        self.debug_logger.debug(f"  Y check: {y1:.0f} < {ball_y} < {y2:.0f} = {y1 < ball_y < y2}")
        self.debug_logger.debug(f"  Result: {'✅ UP detected' if is_in_up_region else '❌ Not in UP region'}")

        return is_in_up_region

    def detect_down_with_selected(self, selected_ball, selected_hoop):
        """
        Detect DOWN state using synchronized selected data from same frame

        Args:
            selected_ball: Ball detection tuple from current frame
            selected_hoop: Hoop detection tuple from current frame

        Returns:
            bool: True if ball is in DOWN region relative to hoop
        """
        if not selected_ball or not selected_hoop:
            return False

        # Ensure same frame (double safety check)
        if selected_ball[1] != selected_hoop[1]:
            print(f"Warning: Frame mismatch in DOWN detection - ball:{selected_ball[1]}, hoop:{selected_hoop[1]}")
            return False

        # Extract positions and dimensions
        ball_y = selected_ball[0][1]
        hoop_y = selected_hoop[0][1]
        hoop_h = selected_hoop[3]

        # Calculate DOWN threshold (using hoop's top edge)
        down_threshold = hoop_y - 0.5 * hoop_h

        is_in_down_region = ball_y > down_threshold

        if is_in_down_region:
            print(f"DOWN detected - Frame {selected_ball[1]}: ball_y({ball_y}) > threshold({down_threshold:.0f}) (hoop top edge)")

        return is_in_down_region

    def should_draw_hoop_and_regions(self):
        """
        Determine if hoop and UP/DOWN regions should be drawn
        Returns tuple: (should_draw, hoop_data, data_source)

        🔧 CRITICAL SYNC FIX: Follow the EXACT same logic as hoop detection rectangle drawing
        Only draw when hoop detection rectangle (cornerRect) is drawn (conf > 0.4)
        """
        hoop_data = None
        data_source = "none"
        should_draw = False

        # 🎯 KEY: Check if current frame has hoop detection that would be drawn
        # This matches the logic in line 889-890: if conf > 0.4: cvzone.cornerRect(...)
        current_frame_hoop_drawn = False
        if hasattr(self, 'current_frame_hoops') and self.current_frame_hoops:
            for hoop in self.current_frame_hoops:
                if hoop.get('confidence', 0) > 0.4:
                    current_frame_hoop_drawn = True
                    break

        # 🔧 CRITICAL FIX: ONLY use current frame hoop data to match cornerRect exactly
        # This ensures perfect alignment with cyan detection rectangles
        if current_frame_hoop_drawn:
            # Find the best hoop that meets drawing criteria (conf > 0.4)
            valid_hoops = [h for h in self.current_frame_hoops if h.get('confidence', 0) > 0.4]
            if valid_hoops:
                best_hoop = max(valid_hoops, key=lambda x: x.get('confidence', 0))
                hoop_data = (
                    (best_hoop['center'][0], best_hoop['center'][1]),
                    self.frame_count,
                    best_hoop['size']['width'],
                    best_hoop['size']['height'],
                    best_hoop['confidence']
                )
                data_source = "current_frame_hoop"
                should_draw = True

                self.debug_logger.debug(f"🎯 PERFECT SYNC: Using current frame hoop conf={best_hoop['confidence']:.2f} center=({best_hoop['center'][0]:.0f},{best_hoop['center'][1]:.0f})")
            else:
                self.debug_logger.debug(f"🎯 PERFECT SYNC: No valid hoops (conf > 0.4) in current frame")
        else:
            self.debug_logger.debug(f"🎯 PERFECT SYNC: No current frame hoops available")

        # 🚫 REMOVED: All other data sources (detection_hoop_data, selected_hoop, trajectory)
        # We ONLY use current frame data to match cornerRect drawing exactly

        return should_draw, hoop_data, data_source

    def draw_detection_regions(self):
        """
        Draw UP and DOWN detection regions on the frame for visualization
        UP region: Orange border
        DOWN region: Purple border

        🔧 SYNC FIX: Only draw regions when hoop should be drawn
        This ensures perfect synchronization between hoop and region visibility
        """
        # Use unified drawing logic
        should_draw, hoop_data, data_source = self.should_draw_hoop_and_regions()

        if not should_draw or not hoop_data:
            self.debug_logger.debug(f"🎨 Not drawing regions - hoop not visible (should_draw={should_draw})")
            return

        # Extract hoop position and dimensions
        hoop_x, hoop_y = hoop_data[0]
        hoop_w, hoop_h = hoop_data[2], hoop_data[3]

        # 🔧 DEBUG: Log the exact hoop data being used for visualization
        self.debug_logger.debug(f"🎨 Drawing regions using {data_source}: Hoop({hoop_x:.0f},{hoop_y:.0f}) {hoop_w}×{hoop_h}")

        # Calculate UP region boundaries (same as detect_up_with_selected)
        up_x1 = int(hoop_x - 3 * hoop_w)
        up_x2 = int(hoop_x + 3 * hoop_w)
        up_y1 = int(hoop_y - 1.5 * hoop_h)
        up_y2 = int(hoop_y - 0.8 * hoop_h)

        # Calculate DOWN region boundaries (simplified visualization)
        down_x1 = int(hoop_x - 1 * hoop_w)
        down_x2 = int(hoop_x + 1 * hoop_w)
        down_y1 = int(hoop_y + 0.5 * hoop_h)  # DOWN threshold line
        down_y2 = int(hoop_y + 2 * hoop_h)    # Extended down for visualization

        # Ensure coordinates are within frame bounds
        frame_h, frame_w = self.frame.shape[:2]
        up_x1 = max(0, min(up_x1, frame_w))
        up_x2 = max(0, min(up_x2, frame_w))
        up_y1 = max(0, min(up_y1, frame_h))
        up_y2 = max(0, min(up_y2, frame_h))

        down_x1 = max(0, min(down_x1, frame_w))
        down_x2 = max(0, min(down_x2, frame_w))
        down_y1 = max(0, min(down_y1, frame_h))
        down_y2 = max(0, min(down_y2, frame_h))

        # Draw UP region with orange border (BGR: 0, 165, 255)
        if up_x2 > up_x1 and up_y2 > up_y1:
            cv2.rectangle(self.frame, (up_x1, up_y1), (up_x2, up_y2), (0, 165, 255), 3)
            # Add label
            cv2.putText(self.frame, "UP", (up_x1 + 5, up_y1 + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)

        # 🔧 MODIFIED: Draw DOWN threshold line at the top of the hoop (rim top)
        # Use the same Y coordinate as detect_down function (hoop_y - 0.5 * hoop_h)
        down_threshold_y = int(hoop_y - 0.5 * hoop_h)  # Rim top position
        if 0 <= down_threshold_y < frame_h:
            # Draw threshold line with purple color (BGR: 128, 0, 128)
            cv2.line(self.frame, (down_x1, down_threshold_y), (down_x2, down_threshold_y), (128, 0, 128), 4)
            # Add label - DOWN text positioned below the line (outside)
            cv2.putText(self.frame, "DOWN", (down_x1 + 5, down_threshold_y + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (128, 0, 128), 2)

        # Draw ball position with a circle if available
        if hasattr(self, 'selected_ball') and self.selected_ball:
            ball_x, ball_y = self.selected_ball[0]
            ball_x, ball_y = int(ball_x), int(ball_y)

            # Draw ball center with a small circle
            cv2.circle(self.frame, (ball_x, ball_y), 8, (0, 255, 0), -1)  # Green filled circle
            cv2.circle(self.frame, (ball_x, ball_y), 12, (255, 255, 255), 2)  # White border

            # Add ball coordinates text
            cv2.putText(self.frame, f"Ball({ball_x},{ball_y})", (ball_x + 15, ball_y - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # 🔧 SYNC FIX: Draw hoop position (synchronized with region drawing)
        hoop_x, hoop_y = int(hoop_x), int(hoop_y)
        cv2.circle(self.frame, (hoop_x, hoop_y), 10, (0, 255, 255), -1)  # Yellow filled circle
        cv2.circle(self.frame, (hoop_x, hoop_y), 15, (255, 255, 255), 2)  # White border

        # Add hoop coordinates text and data source info
        cv2.putText(self.frame, f"Hoop({hoop_x},{hoop_y})", (hoop_x + 20, hoop_y - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # 🔧 DEBUG: Show data source for troubleshooting alignment issues
        cv2.putText(self.frame, f"Source: {data_source}", (hoop_x + 20, hoop_y + 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        self.debug_logger.debug(f"🎨 Synchronized drawing: hoop and regions both visible")

        # Display current UP/DOWN state
        state_text = f"Frame {self.frame_count}: "
        if self.up and self.down:
            state_text += f"UP({self.up_frame}) -> DOWN({self.down_frame})"
            state_color = (0, 255, 255)  # Yellow
        elif self.up:
            state_text += f"UP({self.up_frame})"
            state_color = (0, 165, 255)  # Orange
        elif self.down:
            state_text += f"DOWN({self.down_frame})"
            state_color = (128, 0, 128)  # Purple
        else:
            state_text += "WAITING"
            state_color = (255, 255, 255)  # White

        # Draw state text at top left
        cv2.putText(self.frame, state_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, state_color, 2)

    def shot_detection_with_selected(self):
        """
        Perform UP/DOWN detection using synchronized selected data
        Only called when both ball and hoop are detected in same frame
        """
        if not self.selected_ball or not self.selected_hoop:
            self.debug_logger.debug(f"🚫 Frame {self.frame_count}: Skipping UP/DOWN detection - no synchronized data")
            return

        self.debug_logger.debug(f"🔍 Frame {self.frame_count}: Performing UP/DOWN detection with synchronized data")

        # 🔧 CRITICAL FIX: Save the exact hoop data used for UP/DOWN detection
        # This ensures visualization uses the same data as detection logic
        self.detection_hoop_data = self.selected_hoop

        # UP detection
        if not self.up:
            self.debug_logger.debug(f"🔥 FORCE DEBUG: Attempting UP detection for frame {self.frame_count}")
            up_detected = self.detect_up_with_selected(self.selected_ball, self.selected_hoop)
            self.debug_logger.debug(f"🔥 UP detection result: {up_detected}")
            if up_detected:
                self.up = True
                self.up_frame = self.frame_count  # Use current frame count
                self.debug_logger.info(f"🔥 ✅ UP state detected at frame {self.up_frame}")
            else:
                self.debug_logger.debug(f"🔥 ❌ Frame {self.frame_count}: Ball not in UP region")
        else:
            self.debug_logger.debug(f"🔥 UP already detected at frame {self.up_frame}, skipping UP detection")

        # DOWN detection (only after UP)
        if self.up and not self.down:
            down_detected = self.detect_down_with_selected(self.selected_ball, self.selected_hoop)
            if down_detected:
                self.down = True
                self.down_frame = self.frame_count  # Use current frame count
                self.debug_logger.info(f"✅ DOWN state detected at frame {self.down_frame}")

                # Trigger shot analysis immediately
                self.analyze_shot_attempt()
            else:
                self.debug_logger.debug(f"❌ Frame {self.frame_count}: Ball not in DOWN region")

    def shot_detection(self):
        # Legacy method - kept for compatibility but should not be used
        # New detection uses shot_detection_with_selected()
        pass

    def analyze_shot_attempt(self):
        """Analyze shot attempt when DOWN is detected after UP"""
        # Check if we have enough data to analyze a potential shot
        if len(self.ball_pos) > 0 and len(self.hoop_pos) > 0:
            # Create debug info dictionary
            debug_info = {}

            # Check if this is a valid shot attempt (UP→DOWN sequence)
            is_valid_shot_attempt = (self.up and self.down and self.up_frame < self.down_frame)

            if is_valid_shot_attempt:
                # Valid shot attempt - analyze trajectory
                self.attempts += 1
                self.up = False
                self.down = False

                # Add shot context information
                debug_info['shot_context'] = {
                    'up_frame': self.up_frame,
                    'down_frame': self.down_frame,
                    'frames_between_up_down': self.down_frame - self.up_frame,
                    'total_ball_positions': len(self.ball_pos),
                    'total_hoop_positions': len(self.hoop_pos),
                    'detection_type': 'valid_shot_attempt'
                }

                # Check if it's a make or miss with debug info
                is_successful = score(self.ball_pos, self.hoop_pos, debug_info)

            else:
                # Not a valid shot attempt - record as failed detection
                debug_info['shot_context'] = {
                    'up_frame': self.up_frame if hasattr(self, 'up_frame') else None,
                    'down_frame': self.down_frame if hasattr(self, 'down_frame') else None,
                    'up_detected': self.up,
                    'down_detected': self.down,
                    'total_ball_positions': len(self.ball_pos),
                    'total_hoop_positions': len(self.hoop_pos),
                    'detection_type': 'invalid_shot_attempt'
                }

                # Determine failure reason
                if not self.up and not self.down:
                    debug_info['failure_reason'] = "No UP or DOWN movement detected"
                elif not self.up:
                    debug_info['failure_reason'] = "No UP movement detected (ball didn't enter UP zone)"
                elif not self.down:
                    debug_info['failure_reason'] = "No DOWN movement detected (ball didn't enter DOWN zone)"
                elif self.up_frame >= self.down_frame:
                    debug_info['failure_reason'] = "Invalid sequence: DOWN detected before UP"

                is_successful = False

            # 🔧 CRITICAL FIX: Reset UP/DOWN states after any shot analysis
            # This prevents incorrect UP states from persisting
            self.debug_logger.info(f"🔧 Resetting UP/DOWN states after shot analysis")
            self.debug_logger.debug(f"🔧 Previous states: UP={self.up} (frame {self.up_frame}), DOWN={self.down} (frame {self.down_frame})")
            self.up = False
            self.down = False
            self.up_frame = 0
            self.down_frame = 0
            self.debug_logger.debug(f"🔧 States reset: UP={self.up}, DOWN={self.down}")

            timestamp = self.frame_count / 30  # assuming 30fps

            # Log shot attempt immediately when DOWN is detected
            self.logger.log_shot(
                frame_idx=self.frame_count,
                timestamp=timestamp,
                ball_pos=self.ball_pos[-1][0],
                hoop_pos=self.hoop_pos[-1][0],
                ball_confidence=self.ball_pos[-1][4],  # Use actual ball confidence
                is_successful=is_successful,
                debug_info=debug_info
            )

            # Clear trajectory data after shot analysis is complete
            # This prevents data contamination between different shots
            self.ball_pos = []
            self.hoop_pos = []
            if self.enable_person_detection:
                self.person_pos = []

            if is_successful:
                self.makes += 1
                self.overlay_color = (0, 255, 0)  # Green for make
                self.overlay_text = "Make"
                self.fade_counter = self.fade_frames
            else:
                self.overlay_color = (255, 0, 0)  # Red for miss
                self.overlay_text = "Miss"
                self.fade_counter = self.fade_frames

    def display_score(self):
        # Add text
        text = str(self.makes) + " / " + str(self.attempts)
        cv2.putText(self.frame, text, (50, 125), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 6)
        cv2.putText(self.frame, text, (50, 125), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 3)

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


if __name__ == "__main__":
    import argparse
    from model_configs import get_model_config, list_all_configs

    parser = argparse.ArgumentParser(description='Basketball Shot Detector with Enhanced Model Support')
    parser.add_argument('--input', type=str, default='video_test_5.mp4', help='Input video file path')
    parser.add_argument('--output', type=str, help='Output video file path')
    parser.add_argument('--ball-model', type=str, default='Yolo-Weights/yolov8x.pt', help='Ball detection model (default: yolov8x.pt)')
    parser.add_argument('--hoop-model', type=str, default='Yolo-Weights/best.pt', help='Hoop detection model (default: best.pt)')
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
