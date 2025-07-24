
# Avi Shah - Basketball Shot Detector/Tracker - July 2023

from ultralytics import YOLO
import cv2
import cvzone
import math
import numpy as np
import json
import time
from tqdm import tqdm
from utils import score, detect_down, detect_up, in_hoop_region, clean_hoop_pos, clean_ball_pos, get_device
from datetime import datetime

class ShotLogger:
    def __init__(self, input_video="video_test_5.mp4", ball_threshold=0.5):
        self.shots = []
        self.start_time = time.time()
        self.start_datetime = datetime.now()
        self.frame_count = 0
        self.success_count = 0
        self.total_attempts = 0
        self.progress = 0
        self.input_video = input_video
        self.ball_threshold = ball_threshold

        # 改进的三类投篮记录
        self.successful_shots = []           # 成功投篮
        self.detected_failed_shots = []      # 识别到投篮但失败
        self.undetected_attempts = []        # 未检测到投篮
        
    def log_shot(self, frame_idx, timestamp, ball_pos, hoop_pos, ball_confidence, is_successful, debug_info=None):
        """
        Record shot information with improved classification

        Args:
            frame_idx: Frame index
            timestamp: Timestamp
            ball_pos: Ball position
            hoop_pos: Hoop position
            ball_confidence: Ball confidence
            is_successful: Whether the shot was successful
            debug_info: Debug information dictionary
        """
        if is_successful:
            self.success_count += 1
        self.total_attempts += 1

        # 创建完整的投篮数据
        shot_data = {
            "frame_index": frame_idx,
            "timestamp": timestamp,
            "ball_position": ball_pos,
            "ball_confidence": ball_confidence,
            "hoop_position": hoop_pos,
            "debug_info": debug_info if debug_info else {}
        }

        # 根据debug_info中的detection_type进行分类
        detection_type = debug_info.get('shot_context', {}).get('detection_type', 'unknown') if debug_info else 'unknown'

        if is_successful:
            # 成功投篮
            shot_data["result_category"] = "successful"
            shot_data["is_successful"] = True
            self.successful_shots.append(shot_data)

        elif detection_type == "valid_shot_attempt":
            # 识别到UP→DOWN轨迹但失败
            shot_data["result_category"] = "detected_failed"
            shot_data["is_successful"] = False
            self.detected_failed_shots.append(shot_data)

        else:
            # 未检测到UP→DOWN轨迹
            shot_data["result_category"] = "undetected_attempt"
            shot_data["is_successful"] = False
            self.undetected_attempts.append(shot_data)

        # 保持向后兼容性
        legacy_shot_data = {
            "frame_index": frame_idx,
            "timestamp": timestamp,
            "is_successful": is_successful,
            "_debug_info": debug_info,
            "_ball_position": ball_pos,
            "_hoop_position": hoop_pos,
            "_ball_confidence": ball_confidence
        }
        self.shots.append(legacy_shot_data)
    
    def update_progress(self, current, total):
        self.progress = (current / total) * 100
        
    def save_log(self, filename=None):
        if filename is None:
            # Extract filename from input video path (without extension)
            video_name = self.input_video.split('/')[-1]
            video_name = video_name.split('.')[0]
            # Generate log filename with video name and start time
            timestamp = self.start_datetime.strftime('%Y%m%d_%H%M%S')
            filename = f"{video_name}_shot_log_{timestamp}.json"

        # 生成改进的投篮日志
        processing_time = time.time() - self.start_time

        # 计算统计数据
        total_attempts = len(self.successful_shots) + len(self.detected_failed_shots) + len(self.undetected_attempts)
        successful_count = len(self.successful_shots)
        detected_failed_count = len(self.detected_failed_shots)
        undetected_count = len(self.undetected_attempts)

        # 计算各种成功率
        overall_success_rate = (successful_count / total_attempts * 100) if total_attempts > 0 else 0
        valid_attempts = successful_count + detected_failed_count
        shooting_accuracy = (successful_count / valid_attempts * 100) if valid_attempts > 0 else 0
        detection_accuracy = (valid_attempts / total_attempts * 100) if total_attempts > 0 else 0

        # 分析失败原因
        failure_analysis = self._analyze_failures()

        improved_log = {
            "input_video": self.input_video,
            "processing_start": self.start_datetime.strftime("%Y-%m-%d %H:%M:%S"),
            "processing_time_seconds": round(processing_time, 2),
            "total_frames": self.frame_count,
            "ball_threshold": self.ball_threshold,

            # 总体统计
            "summary": {
                "total_attempts": total_attempts,
                "successful_shots_count": successful_count,
                "detected_failed_shots_count": detected_failed_count,
                "undetected_attempts_count": undetected_count,

                # 各种成功率
                "overall_success_rate": round(overall_success_rate, 2),
                "shooting_accuracy": round(shooting_accuracy, 2),  # 在有效投篮中的成功率
                "detection_accuracy": round(detection_accuracy, 2),  # 投篮检测准确率
            },

            # 分类详情
            "shot_categories": {
                "successful_shots": {
                    "count": successful_count,
                    "description": "成功投篮 - 检测到UP→DOWN轨迹且球进筐",
                    "shots": self.successful_shots
                },
                "detected_failed_shots": {
                    "count": detected_failed_count,
                    "description": "识别到UP→DOWN轨迹但失败 - 球未进筐",
                    "shots": self.detected_failed_shots
                },
                "undetected_attempts": {
                    "count": undetected_count,
                    "description": "未检测到UP→DOWN轨迹 - 不符合投篮模式",
                    "shots": self.undetected_attempts
                }
            },

            # 失败原因分析
            "failure_analysis": failure_analysis
        }

        # 保存改进的日志
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(improved_log, f, indent=2, ensure_ascii=False)

        return filename

    def _analyze_failures(self):
        """分析失败原因"""
        failure_reasons = {}

        # 分析检测到的失败投篮
        for shot in self.detected_failed_shots:
            reason = shot.get('debug_info', {}).get('failure_reason', 'Unknown')
            if reason not in failure_reasons:
                failure_reasons[reason] = {"detected_failed": 0, "undetected": 0}
            failure_reasons[reason]["detected_failed"] += 1

        # 分析未检测到的投篮
        for shot in self.undetected_attempts:
            reason = shot.get('debug_info', {}).get('failure_reason', 'Unknown')
            if reason not in failure_reasons:
                failure_reasons[reason] = {"detected_failed": 0, "undetected": 0}
            failure_reasons[reason]["undetected"] += 1

        return failure_reasons



    def print_improved_summary(self):
        """打印改进的摘要"""
        total_attempts = len(self.successful_shots) + len(self.detected_failed_shots) + len(self.undetected_attempts)
        successful_count = len(self.successful_shots)
        detected_failed_count = len(self.detected_failed_shots)
        undetected_count = len(self.undetected_attempts)

        # 计算各种成功率
        overall_success_rate = (successful_count / total_attempts * 100) if total_attempts > 0 else 0
        valid_attempts = successful_count + detected_failed_count
        shooting_accuracy = (successful_count / valid_attempts * 100) if valid_attempts > 0 else 0
        detection_accuracy = (valid_attempts / total_attempts * 100) if total_attempts > 0 else 0

        print("\n" + "="*60)
        print("📊 改进的投篮检测报告")
        print("="*60)

        print(f"🎥 视频: {self.input_video}")
        print(f"⏱️  处理时间: {time.time() - self.start_time:.2f}秒")
        print(f"🎯 球检测阈值: {self.ball_threshold}")

        print(f"\n📈 总体统计:")
        print(f"  总尝试次数: {total_attempts}")
        print(f"  成功投篮: {successful_count}")
        print(f"  识别到但失败: {detected_failed_count}")
        print(f"  未检测到: {undetected_count}")

        print(f"\n🎯 成功率分析:")
        print(f"  总体成功率: {overall_success_rate:.1f}%")
        print(f"  投篮准确率: {shooting_accuracy:.1f}% (在有效投篮中)")
        print(f"  检测准确率: {detection_accuracy:.1f}% (正确识别投篮)")

        print(f"\n📋 分类详情:")
        print(f"  成功投篮 - 检测到UP→DOWN轨迹且球进筐: {successful_count}次")
        print(f"  识别到UP→DOWN轨迹但失败 - 球未进筐: {detected_failed_count}次")
        print(f"  未检测到UP→DOWN轨迹 - 不符合投篮模式: {undetected_count}次")

        # 显示失败原因
        failure_analysis = self._analyze_failures()
        if failure_analysis:
            print(f"\n❌ 失败原因分析:")
            for reason, counts in failure_analysis.items():
                total = counts["detected_failed"] + counts["undetected"]
                print(f"  {reason}: {total}次")
                if counts["detected_failed"] > 0:
                    print(f"    - 识别到但失败: {counts['detected_failed']}次")
                if counts["undetected"] > 0:
                    print(f"    - 未检测到: {counts['undetected']}次")

    def log_frame_data(self, frame_count, all_balls, all_hoops, all_persons=None, selected_ball_idx=-1, selected_hoop_idx=-1, selected_person_idx=-1,
                       current_frame_balls=None, current_frame_hoops=None, current_frame_persons=None):
        """
        Log detailed frame processing data for analysis

        Args:
            frame_count: Current frame number
            all_balls: List of all ball trajectory points (historical data)
            all_hoops: List of all hoop trajectory points (historical data)
            all_persons: List of all person trajectory points (historical data)
            selected_ball_idx: Index of selected ball in all_balls (-1 if none)
            selected_hoop_idx: Index of selected hoop in all_hoops (-1 if none)
            selected_person_idx: Index of selected person in all_persons (-1 if none)
            current_frame_balls: List of all balls detected by YOLO in current frame
            current_frame_hoops: List of all hoops detected by YOLO in current frame
            current_frame_persons: List of all persons detected by YOLO in current frame
        """
        # Skip logging if no debug file is being created
        if not hasattr(self, '_frame_log_file'):
            # Create frame log filename based on input video
            video_name = self.input_video.split('/')[-1].split('.')[0]
            timestamp = self.start_datetime.strftime('%Y%m%d_%H%M%S')
            frame_log_filename = f"{video_name}_frame_log_{timestamp}.json"
            self._frame_log_file = open(frame_log_filename, 'w')
            self._frame_log_file.write('[')  # Start JSON array
            self._first_frame_logged = False  # Track if first frame has been logged

        # Prepare frame data
        frame_data = {
            "frame": frame_count,
            "timestamp": frame_count / 30.0,  # Assuming 30fps
            "ball_threshold": 0.4,  # Current ball confidence threshold
            "hoop_threshold": 0.4,  # Current hoop confidence threshold
            "person_threshold": 0.3,  # Current person confidence threshold
            "trajectory_balls": [],  # Historical ball trajectory points
            "trajectory_hoops": [],  # Historical hoop trajectory points
            "trajectory_persons": [],  # Historical person trajectory points
            "current_detections": {  # All YOLO detections in current frame
                "balls": current_frame_balls if current_frame_balls else [],
                "hoops": current_frame_hoops if current_frame_hoops else [],
                "persons": current_frame_persons if current_frame_persons else []
            },
            "selected_ball_idx": selected_ball_idx,
            "selected_hoop_idx": selected_hoop_idx,
            "selected_person_idx": selected_person_idx,
            "selected_ball": all_balls[selected_ball_idx] if selected_ball_idx >= 0 else None,
            "selected_hoop": all_hoops[selected_hoop_idx] if selected_hoop_idx >= 0 else None,
            "selected_person": all_persons[selected_person_idx] if all_persons and selected_person_idx >= 0 else None
        }

        # Add all trajectory ball points (historical data)
        for i, ball in enumerate(all_balls):
            frame_data["trajectory_balls"].append({
                "index": i,
                "position": ball[0],
                "frame": ball[1],
                "confidence": float(ball[4]),
                "size": {"width": ball[2], "height": ball[3]},
                "above_threshold": float(ball[4]) >= 0.2
            })

        # Add all trajectory hoop points (historical data)
        for i, hoop in enumerate(all_hoops):
            frame_data["trajectory_hoops"].append({
                "index": i,
                "position": hoop[0],
                "frame": hoop[1],
                "confidence": float(hoop[4]),
                "size": {"width": hoop[2], "height": hoop[3]},
                "above_threshold": float(hoop[4]) >= 0.4
            })

        # Add all trajectory person points (historical data)
        if all_persons:
            for i, person in enumerate(all_persons):
                frame_data["trajectory_persons"].append({
                    "index": i,
                    "position": person[0],
                    "frame": person[1],
                    "confidence": float(person[4]),
                    "size": {"width": person[2], "height": person[3]},
                    "above_threshold": float(person[4]) >= 0.3
                })

        # Write frame data to frame log file
        if self._first_frame_logged:
            self._frame_log_file.write(',\n')
        json.dump(frame_data, self._frame_log_file, indent=2)
        self._first_frame_logged = True


        


class ShotDetector:
    def __init__(self, input_video="video_test_5.mp4", output_video=None, ball_model_path="yolov8m.pt", hoop_model_path="best.pt", person_model_path=None, use_shared_model=True, min_ball_area=400, enable_person_detection=False):
        # Load models for optimal detection
        self.overlay_text = "Waiting..."
        self.use_shared_model = use_shared_model
        self.enable_person_detection = enable_person_detection

        # Load main detection model (YOLOv8m for sports ball and person)
        self.ball_model_path = ball_model_path
        self.ball_model = YOLO(ball_model_path)
        print(f"🏀 Loaded main model: {ball_model_path}")

        # Load hoop detection model (custom trained model)
        self.hoop_model_path = hoop_model_path
        self.hoop_model = YOLO(hoop_model_path)
        print(f"🏀 Loaded hoop model: {hoop_model_path}")

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
        self.output_video = output_video
        self.video_writer = None
        self.input_video = input_video
        self.logger = ShotLogger(input_video=input_video, ball_threshold=0.5)
        
        # Uncomment this line to accelerate inference. Note that this may cause errors in some setups.
        #self.model.half()
        
        # Initialize class names for both models
        self.class_names = ['Basketball', 'Basketball Hoop']  # Default for custom models

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
                           if 'hoop' in cls.lower() or cls.lower() == 'basketball hoop']
            print(f"📋 Hoop model classes: {list(self.hoop_model_classes.values())}")
            print(f"🎯 Active hoop detection classes: {hoop_classes}")
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

        self.frame_count = 0
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

                            # Filter out balls smaller than minimum area and apply confidence/region checks
                            if ball_area >= self.min_ball_area and (conf > 0.2 or (in_hoop_region(center, self.hoop_pos) and conf > 0.1)):
                                self.ball_pos.append((center, self.frame_count, w, h, conf))
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

                        # Check if this is a basketball hoop
                        is_hoop = (current_class in ["Basketball Hoop", "hoop"] or
                                  "hoop" in current_class.lower() or
                                  (current_class.lower() == "basketball hoop"))

                        if is_hoop:
                            current_frame_hoops.append({
                                "bbox": [x1, y1, x2, y2],
                                "center": center,
                                "size": {"width": w, "height": h},
                                "confidence": float(conf),
                                "class": current_class
                            })

                            # Create hoop points if high confidence
                            if conf > 0.4:
                                self.hoop_pos.append((center, self.frame_count, w, h, conf))
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

            self.clean_motion()
            self.shot_detection()
            self.display_score()
            self.frame_count += 1
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

            # Determine selected indices (default to last detected if any)
            selected_ball_idx = len(all_balls) - 1 if all_balls else -1
            selected_hoop_idx = len(all_hoops) - 1 if all_hoops else -1
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
                current_persons_data
            )

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

    def clean_motion(self):
        # Clean and display ball motion
        self.ball_pos = clean_ball_pos(self.ball_pos, self.frame_count)
        for i in range(0, len(self.ball_pos)):
            cv2.circle(self.frame, self.ball_pos[i][0], 2, (0, 0, 255), 2)

        # Clean hoop motion and display current hoop center
        if len(self.hoop_pos) > 1:
            self.hoop_pos = clean_hoop_pos(self.hoop_pos)

        # Draw hoop center if hoop positions exist
        if len(self.hoop_pos) > 0:
            cv2.circle(self.frame, self.hoop_pos[-1][0], 2, (128, 128, 0), 2)

    def shot_detection(self):
        if len(self.hoop_pos) > 0 and len(self.ball_pos) > 0:
            # Detecting when ball is in 'up' and 'down' area - ball can only be in 'down' area after it is in 'up'
            if not self.up:
                self.up = detect_up(self.ball_pos, self.hoop_pos)
                if self.up:
                    self.up_frame = self.ball_pos[-1][1]

            if self.up and not self.down:
                self.down = detect_down(self.ball_pos, self.hoop_pos)
                if self.down:
                    self.down_frame = self.ball_pos[-1][1]

            # Check for shot detection every 10 frames to avoid duplicate detections
            # This ensures each shot attempt is recorded only once
            if self.frame_count % 10 == 0:
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

                    timestamp = self.frame_count / 30  # assuming 30fps

                    # Log every detection attempt (both valid shots and failed detections) every 10 frames
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
    parser.add_argument('--ball-model', type=str, default='yolov8m.pt', help='Ball detection model (default: yolov8m.pt)')
    parser.add_argument('--hoop-model', type=str, default='best.pt', help='Hoop detection model (default: best.pt)')
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
