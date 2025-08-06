# Avi Shah - Basketball Shot Detector/Tracker - July 2023

from ultralytics import YOLO
import cv2
import cvzone
import math
import numpy as np
import json
import time
from tqdm import tqdm
import torch
# Try to import PySceneDetect - if not available, we'll handle it gracefully
try:
    from scenedetect import detect, ContentDetector
    SCENE_DETECTION_AVAILABLE = True
except ImportError:
    SCENE_DETECTION_AVAILABLE = False
    print("Warning: PySceneDetect not available. Scene change detection will be disabled.")

from utils import score, detect_down, detect_up, in_hoop_region, clean_hoop_pos, select_ball, get_device
from datetime import datetime
import os

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
        self._first_frame_logged = False  # Add missing attribute
        self.output_dir = None  # Store output directory
        self.model_name = None  # Store model name for consistent naming
        # Store scene change detection results
        self.scene_changes = []
        # Debug log file
        self._debug_log_file = None
        self._debug_log_first_entry = False
        
    def set_output_info(self, output_dir, model_name):
        """Set output directory and model name for consistent log naming"""
        self.output_dir = output_dir
        self.model_name = model_name
        
        # Create debug log file immediately after setting output info
        try:
            # Create debug log filename based on input video
            video_name = os.path.splitext(os.path.basename(self.input_video))[0]
            timestamp = self.start_datetime.strftime('%Y-%m-%d_%H-%M-%S')
            
            # Use consistent naming with model name if available
            if self.output_dir and self.model_name:
                debug_filename = os.path.join(self.output_dir, f"{video_name}_{self.model_name}_debug_{timestamp}.txt")
            else:
                debug_filename = f"{video_name}_debug_{timestamp}.txt"
                
            # Create output directory if it doesn't exist
            if self.output_dir:
                os.makedirs(self.output_dir, exist_ok=True)
                
            # Create the debug log file
            self._debug_log_file = open(debug_filename, 'w')
            
            # Add initial test message to verify logging is working
            self.debug_log("Debug logging initialized successfully")
            
        except Exception as e:
            print(f"Error creating debug log file: {e}")
            self._debug_log_file = None
    
    def log_scene_changes(self, scene_changes):
        """Log scene change detection results"""
        self.scene_changes = scene_changes
        
    def log_shot(self, frame_idx, timestamp, ball_pos, hoop_pos, ball_confidence, is_successful, debug_info=None):
        """
        Record shot information, with minimal data in shot log and detailed info in debug log
        
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
        
        # Store only essential information in shot log
        shot_data = {
            "frame_index": frame_idx,
            "timestamp": timestamp,
            "is_successful": is_successful,
            "debug_log_id": self.total_attempts  # Reference to debug log
        }
        
        # Store the debug info separately for debug log
        if debug_info:
            shot_data["_debug_info"] = debug_info  # Temporary storage, won't be included in final shot log
            shot_data["_ball_position"] = ball_pos  # Temporary storage
            shot_data["_hoop_position"] = hoop_pos  # Temporary storage
            shot_data["_ball_confidence"] = ball_confidence  # Temporary storage
            
        self.shots.append(shot_data)
    
    def update_progress(self, current, total):
        self.progress = (current / total) * 100
        
    def save_log(self, filename=None):
        if filename is None:
            # Extract filename from input video path (without extension)
            video_name = os.path.splitext(os.path.basename(self.input_video))[0]
            
            # Generate log filename with video name, model name and start time
            timestamp = self.start_datetime.strftime('%Y-%m-%d_%H-%M-%S')
            
            # Use consistent naming format with model name
            if self.output_dir and self.model_name:
                filename = os.path.join(self.output_dir, f"{video_name}_{self.model_name}_shot_{timestamp}.json")
            else:
                filename = f"{video_name}_shot_{timestamp}.json"
        
        # Create output directory if needed
        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)
        
        # Create a clean copy of shots without debug info
        clean_shots = []
        for shot in self.shots:
            clean_shot = {k: v for k, v in shot.items() if not k.startswith('_')}
            clean_shots.append(clean_shot)
        
        stats = {
            "input_video": self.input_video,
            "processing_start": datetime.fromtimestamp(self.start_time).strftime('%Y-%m-%d %H:%M:%S'),
            "total_frames": self.frame_count,
            "total_attempts": self.total_attempts,
            "successful_shots": self.success_count,
            "success_rate": round(self.success_count / self.total_attempts * 100, 2) if self.total_attempts > 0 else 0,
            "processing_time_seconds": round(time.time() - self.start_time, 2),
            "ball_threshold": self.ball_threshold,
            "scene_changes": self.scene_changes,  # Add scene changes to stats
            "shots": clean_shots
        }
        with open(filename, 'w') as f:
            json.dump(stats, f, indent=2)
        
        # Generate detailed debug log
        self.save_debug_log(filename)
        
        # Close debug log file if it exists
        self.close_debug_log()
        
        return filename
        
    def log_frame_data(self, frame_count, all_balls, all_hoops, selected_ball_idx=-1, selected_hoop_idx=-1,
                       current_frame_balls=None, current_frame_hoops=None):
        """
        Log detailed frame processing data for analysis

        Args:
            frame_count: Current frame number
            all_balls: List of all ball trajectory points (historical data)
            all_hoops: List of all hoop trajectory points (historical data)
            selected_ball_idx: Index of selected ball in all_balls (-1 if none)
            selected_hoop_idx: Index of selected hoop in all_hoops (-1 if none)
            current_frame_balls: List of all balls detected by YOLO in current frame
            current_frame_hoops: List of all hoops detected by YOLO in current frame
        """
        # Skip logging if no debug file is being created
        if not hasattr(self, '_debug_file'):
            # Create debug log filename based on input video
            video_name = os.path.splitext(os.path.basename(self.input_video))[0]
            timestamp = self.start_datetime.strftime('%Y-%m-%d_%H-%M-%S')
            
            # Use consistent naming with model name if available
            if self.output_dir and self.model_name:
                frame_filename = os.path.join(self.output_dir, f"{video_name}_{self.model_name}_frame_{timestamp}.json")
            else:
                frame_filename = f"{video_name}_frame_log_{timestamp}.json"
                
            self._debug_file = open(frame_filename, 'w')
            self._debug_file.write('[')  # Start JSON array
            self._first_frame_logged = False  # Track if first frame has been logged
        
        # Prepare frame data
        frame_data = {
            "frame": frame_count,
            "timestamp": frame_count / 30.0,  # Assuming 30fps
            "ball_threshold": 0.2,  # Current ball confidence threshold
            "hoop_threshold": 0.6,  # Current hoop confidence threshold
            "trajectory_balls": [],  # Historical ball trajectory points
            "trajectory_hoops": [],  # Historical hoop trajectory points
            "current_detections": {  # All YOLO detections in current frame
                "balls": current_frame_balls if current_frame_balls else [],
                "hoops": current_frame_hoops if current_frame_hoops else []
            },
            "selected_ball_idx": selected_ball_idx,
            "selected_hoop_idx": selected_hoop_idx,
            "selected_ball": all_balls[selected_ball_idx] if selected_ball_idx >= 0 else None,
            "selected_hoop": all_hoops[selected_hoop_idx] if selected_hoop_idx >= 0 else None
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
                "above_threshold": float(hoop[4]) >= 0.6
            })
        
        # Write frame data to debug file
        if self._first_frame_logged:
            self._debug_file.write(',\n')
        json.dump(frame_data, self._debug_file, indent=2)
        self._first_frame_logged = True

    def save_debug_log(self, shot_log_path):
        """
        Save detailed debug log with comprehensive shot information
        """
        # Generate debug log filename based on shot log path
        base_name = os.path.splitext(shot_log_path)[0]
        
        # Extract the video name and model name part (before "_shot_")
        if "_shot_" in base_name:
            # Split on "_shot_" and take the first part
            parts = base_name.split("_shot_")
            debug_log_path = f"{parts[0]}_debug.json"
        else:
            # Fallback: just remove "_shot" if present
            debug_log_path = base_name.replace("_shot", "") + "_debug.json"
        
        # Read the existing shot log
        try:
            with open(shot_log_path, 'r') as f:
                shot_log_data = json.load(f)
        except Exception as e:
            print(f"Error reading shot log: {e}")
            return shot_log_path
            
        # Add detailed shot information to the shot log
        detailed_shots = []
        for shot in self.shots:
            detailed_shot = {
                "frame_index": shot["frame_index"],
                "timestamp": shot["timestamp"],
                "is_successful": shot["is_successful"]
            }
            
            # Add detailed debug information if available
            if "_debug_info" in shot:
                debug_info = shot["_debug_info"]
            
                # Replace success/failure reason with English versions
                if "success_reason" in debug_info:
                    original_reason = debug_info["success_reason"]
                    if "篮筐中心" in original_reason:
                        debug_info["success_reason"] = "Ball passed through the center of the hoop"
                    elif "篮筐区域" in original_reason:
                        debug_info["success_reason"] = "Ball passed through the hoop area"
                    # Add more translations as needed
        
                if "failure_reason" in debug_info:
                    original_reason = debug_info["failure_reason"]
                    if "未通过篮筐" in original_reason:
                        debug_info["failure_reason"] = "Ball did not pass through the hoop"
                    elif "未检测到球" in original_reason:
                        debug_info["failure_reason"] = "Ball was not detected"
                    elif "置信度低" in original_reason:
                        debug_info["failure_reason"] = "Low confidence in ball detection"
                    # Add more translations as needed
            
                # Add ball and hoop tracking data if available
                if "ball_tracking" in debug_info:
                    detailed_shot["ball_tracking"] = debug_info["ball_tracking"]
            
                if "hoop_tracking" in debug_info:
                    detailed_shot["hoop_tracking"] = debug_info["hoop_tracking"]
            
                # Add other debug info
                detailed_shot["debug_info"] = {k: v for k, v in debug_info.items() 
                                              if k not in ["ball_tracking", "hoop_tracking"]}
                
                # Add concise result reason
                if shot["is_successful"] and "success_reason" in debug_info:
                    detailed_shot["result_reason"] = debug_info["success_reason"]
                elif not shot["is_successful"] and "failure_reason" in debug_info:
                    detailed_shot["result_reason"] = debug_info["failure_reason"]
            
            detailed_shots.append(detailed_shot)
        
        # Add detailed shots to the main log
        shot_log_data["detailed_shots"] = detailed_shots
        
        # Save updated shot log with detailed information
        with open(shot_log_path, 'w') as f:
            json.dump(shot_log_data, f, indent=2)
        
        print(f"Shot log with details saved to {shot_log_path}")
        return shot_log_path

    def debug_log(self, message):
        """
        Log debug message to debug log file with consistent naming
        
        Args:
            message: Debug message to log (string or dict)
        """
        # Check if debug log file exists
        if not self._debug_log_file:
            print("Debug log file not available. Message not logged:", message)
            return
            
        try:
            # Write debug entry to file with timestamp
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]  # Include milliseconds
            if isinstance(message, dict):
                # Convert dict to string for txt format
                message_str = json.dumps(message, indent=2)
            else:
                message_str = str(message)
                
            self._debug_log_file.write(f"[{timestamp}] {message_str}\n")
            
            # Flush to ensure data is written
            self._debug_log_file.flush()
        except Exception as e:
            print(f"Error writing to debug log: {e}")
        
    def close_debug_log(self):
        """
        Close debug log file if it exists
        """
        if self._debug_log_file:
            try:
                self._debug_log_file.close()
            except Exception as e:
                print(f"Error closing debug log file: {e}")
            self._debug_log_file = None
class ShotDetector:
    def __init__(self, input_video="video_test_5.mp4", output_video=None, model_path="best.pt", 
                 ball_model_path=None, hoop_model_path=None, person_model_path=None, use_shared_model=True, 
                 min_ball_area=400, enable_person_detection=False, model_config=None, debug_log_path=None, 
                 output_dir=None, ball_conf_threshold=0.5):
        # For compatibility with batch_test_evaluator.py
        # Use ball_model_path if provided, otherwise fall back to model_path
        actual_model_path = ball_model_path if ball_model_path else model_path
        
        # Initialize logger first
        self.logger = ShotLogger(input_video, ball_conf_threshold)
        
        # Load the YOLO model created from main.py - change text to your relative path
        self.overlay_text = "Waiting..."
        self.model = YOLO(actual_model_path)
        self.model_path = actual_model_path
        self.output_video = output_video
        self.video_writer = None
        self.input_video = input_video
        self.output_dir = output_dir  # Store output directory
        
        # Set device based on CUDA availability
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        try:
            self.model.to(self.device)
        except Exception as e:
            print(f"Warning: Could not move model to {self.device}: {e}")
            self.device = 'cpu'
            self.model.to(self.device)
        
        # Set up debug log path
        self.debug_log_path = debug_log_path or os.path.join('logs', 'console_output.log')
        
        # Pass output directory and model name to logger for consistent naming
        model_name = os.path.splitext(os.path.basename(actual_model_path))[0]
        self.logger.set_output_info(output_dir, model_name)
        
        # Setup class names
        self.setup_class_names()
        
        # Log initialization
        self.logger.debug_log(f"ShotDetector initialized with model: {actual_model_path}")
        
        # Uncomment this line to accelerate inference. Note that this may cause errors in some setups.
        #self.model.half()
        
        # Set device based on availability
        self.device = get_device()
        
        self.device = get_device()
        # Uncomment line below to use webcam (I streamed to my iPhone using Iriun Webcam)
        # self.cap = cv2.VideoCapture(0)

        # Use video from input parameter
        self.cap = cv2.VideoCapture(input_video)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Log video info
        self.logger.debug_log(f"Video loaded: {input_video}, Total frames: {self.total_frames}")

        self.ball_pos = []  # array of tuples ((x_pos, y_pos), frame count, width, height, conf)
        self.hoop_pos = []  # array of tuples ((x_pos, y_pos), frame count, width, height, conf)

        self.frame_count = 0
        self.frame = None

        self.makes = 0
        self.attempts = 0

        # Used to detect shots (upper and lower region)
        self.up = False
        self.down = False
        self.up_frame = 0
        self.down_frame = 0
        self.up_hoop_frame = 0
        self.down_hoop_frame = 0

        # Used for green and red colors after make/miss
        self.fade_frames = 20
        self.fade_counter = 0
        self.overlay_color = (0, 0, 0)

        # Scene change detection using PySceneDetect
        self.scene_changes = []
        self.next_scene_frame = None
        self.scene_change_threshold = 40.0  # Default threshold for scene change detection
        self._detect_scene_changes()
        
        # Setup video writer if output path is provided
        if output_video:
            # Get video properties
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
        
    def setup_class_names(self):
        """
        Set up class names based on the model being used
        """
        # Get class names from the model
        if hasattr(self.model, 'names'):
            self.class_names = self.model.names
        else:
            # Default class names if not available from model
            self.class_names = {0: 'ball', 1: 'hoop'}
        
        print(f"Model: {self.model_path}")
        print(f"Class names: {self.class_names}")
        self.logger.debug_log(f"Model class names: {self.class_names}")
            
    def _detect_scene_changes(self):
        """
        Detect scene changes in the video using PySceneDetect
        """
        if not SCENE_DETECTION_AVAILABLE:
            print("PySceneDetect not available. Skipping scene change detection.")
            self.logger.debug_log("PySceneDetect not available. Skipping scene change detection.")
            return
            
        try:
            # Detect scene changes using ContentDetector with a specific threshold
            scene_list = detect(self.input_video, ContentDetector(threshold=self.scene_change_threshold))
            
            # Convert scene boundaries to frame numbers
            scene_frames = []
            for scene_start, scene_end in scene_list:
                # We want to trigger shot detection before the scene change
                # So we use the frame just before the scene change
                scene_frame = scene_start.get_frames() - 1
                # Skip first and last frames as requested
                if scene_frame > 0 and scene_frame < self.total_frames - 1:
                    scene_frames.append(scene_frame)
            
            self.scene_changes = scene_frames
            self.logger.log_scene_changes([{
                "frame": frame,
                "confidence": self.scene_change_threshold  # Using threshold as confidence measure
            } for frame in scene_frames])
            
            if scene_frames:
                self.next_scene_frame = scene_frames[0]
            else:
                self.next_scene_frame = None
                
            print(f"Detected {len(scene_frames)} scene changes at frames: {scene_frames} (threshold: {self.scene_change_threshold})")
            self.logger.debug_log(f"Detected {len(scene_frames)} scene changes at frames: {scene_frames} (threshold: {self.scene_change_threshold})")
        except Exception as e:
            print(f"Error detecting scene changes: {e}")
            self.logger.debug_log(f"Error detecting scene changes: {e}")
            self.scene_changes = []
            self.next_scene_frame = None

    def run(self):
        self.cap = cv2.VideoCapture(self.input_video)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Log initialization info
        self.logger.debug_log(f"Model class names: {self.class_names}")
        self.logger.debug_log(f"ShotDetector initialized with model: {self.model_path}")
        
        # Detect scene changes if PySceneDetect is available
        if SCENE_DETECTION_AVAILABLE:
            try:
                scene_list = detect(self.input_video, ContentDetector(threshold=40.0))
                self.scene_changes = [scene[0].get_frames() for scene in scene_list]
                self.logger.log_scene_changes(self.scene_changes)
                self.logger.debug_log(f"Detected {len(self.scene_changes)} scene changes at frames: {self.scene_changes} (threshold: 40.0)")
            except Exception as e:
                self.logger.debug_log(f"Scene detection failed: {str(e)}")
                self.scene_changes = []
        else:
            self.logger.debug_log("PySceneDetect not available, scene change detection disabled")
            self.scene_changes = []
        
        # Progress bar
        progress_bar = tqdm(total=self.total_frames, desc="Processing Video", unit="frame")
        
        # Create output directory if it doesn't exist
        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)
            
        # Process video with YOLO model
        # Use CUDA if available, otherwise use CPU
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.logger.debug_log(f"Using device: {device}")
        try:
            results = self.model(
                self.input_video, 
                stream=True, 
                device=device,  # Use CUDA if available, otherwise CPU
                verbose=False  # Reduce verbose output
            )
        except NotImplementedError as e:
            # Fallback to CPU if CUDA is not supported for some operations
            self.logger.debug_log(f"CUDA not supported for this operation, falling back to CPU: {str(e)}")
            device = 'cpu'
            results = self.model(
                self.input_video, 
                stream=True, 
                device=device,  # Force CPU as fallback
                verbose=False  # Reduce verbose output
            )
        
        while self.cap.isOpened():
            success, self.frame = self.cap.read()
            if not success:
                break

            # Run model on the frame to detect objects
            results = self.model(self.frame, stream=True, verbose=False)

            current_frame_balls = []
            current_frame_hoops = []

            # Process detection results
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    # Bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    w, h = x2 - x1, y2 - y1

                    # Confidence and class
                    conf = math.ceil((box.conf[0] * 100)) / 100
                    cls = int(box.cls[0])
                    class_name = self.class_names[cls] if cls in self.class_names else 'unknown'

                    center = (int(x1 + w / 2), int(y1 + h / 2))

                    # Create detection data
                    detection_data = {
                        'bbox': [x1, y1, x2, y2],
                        'center': center,
                        'size': {'width': w, 'height': h},
                        'confidence': conf,
                        'class': class_name,
                        'frame': self.frame_count
                    }

                    # Classify detected object
                    if class_name.lower() in ['ball', 'sports ball']:
                        # Ball detection
                        current_frame_balls.append(detection_data)
                    
                    elif class_name.lower() in ['hoop', 'basketball hoop', 'rim']:
                        # Hoop detection
                        current_frame_hoops.append(detection_data)

            # Add ball and hoop positions to tracking arrays
            # Select the best ball from current frame detections
            selected_ball = select_ball(self.ball_pos, current_frame_balls, 0.5)
            
            # If a suitable ball was selected, add it to tracking
            if selected_ball is not None:
                center = selected_ball['center']
                w = selected_ball['size']['width']
                h = selected_ball['size']['height']
                conf = selected_ball['confidence']
                self.ball_pos.append((center, self.frame_count, w, h, conf))
            
            # Add high confidence hoops to tracking
            for hoop in current_frame_hoops:
                if hoop['confidence'] > 0.5:
                    center = hoop['center']
                    w = hoop['size']['width']
                    h = hoop['size']['height']
                    conf = hoop['confidence']
                    self.hoop_pos.append((center, self.frame_count, w, h, conf))
            
            # Draw detected objects
            self.draw_detections(current_frame_balls, selected_ball, current_frame_hoops)

            
            # Log frame data before incrementing frame count
            all_balls = self.ball_pos if hasattr(self, 'ball_pos') else []
            all_hoops = self.hoop_pos if hasattr(self, 'hoop_pos') else []
            
            # Determine selected indices (default to last detected if any)
            selected_ball_idx = len(all_balls) - 1 if all_balls else -1
            selected_hoop_idx = len(all_hoops) - 1 if all_hoops else -1
            
            self.logger.log_frame_data(
                self.frame_count,
                all_balls,
                all_hoops,
                selected_ball_idx,
                selected_hoop_idx,
                current_frame_balls,
                current_frame_hoops
            )

            self.clean_motion()
            self.shot_detection()
            self.display_score()
            
            # Increment frame count after logging
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

        progress_bar.close()
        self.cap.release()
        if self.video_writer:
            self.video_writer.release()
        else:
            cv2.destroyAllWindows()
        
        # Close frame log file if it exists
        if hasattr(self.logger, '_debug_file'):
            self.logger._debug_file.write('\n]')  # Close JSON array
            self.logger._debug_file.close()
        
        # At the end of processing, save the logs
        shot_log_file = self.logger.save_log()
        self.logger.debug_log(f"Processing completed. Shot log saved to: {shot_log_file}")
        
        # Close all files
        if self.video_writer:
            self.video_writer.release()
        cv2.destroyAllWindows()
        
        return shot_log_file  # Return log filename for batch evaluator

    def draw_detections(self, current_frame_balls, selected_ball, current_frame_hoops):
        """Draw all detected objects with appropriate colors and labels"""
        # Object type configuration with enhanced visibility
        obj_configs = {
            'ball': {
                'color': (0, 255, 0),  # Green
                'selected_color': (255, 0, 0),  # Blue
                'label': 'Ball'
            },
            'selected_ball': {
                'color': (255, 0, 0),  # Red for selected ball
                'label': 'Selected Ball'
            },
            'hoop': {
                'color': (0, 0, 255),  # Red
                'label': 'Hoop'
            }
        }
        
        # Draw all balls
        for ball in current_frame_balls:
            x1, y1 = ball['bbox'][0], ball['bbox'][1]
            w, h = ball['size']['width'], ball['size']['height']
            conf = ball['confidence']
            
            # Determine if this is the selected ball
            if selected_ball is not None and ball == selected_ball:
                config = obj_configs['selected_ball']
            else:
                config = obj_configs['ball']
            
            # Draw the detection box with thicker lines for better visibility
            cvzone.cornerRect(self.frame, (x1, y1, w, h), 
                             colorR=config['color'], 
                             colorC=config['color'],
                             t=3,     # Thickness of the rectangle
                             rt=2)    # Thickness of corner rectangles
            
            # Draw label with background for better readability
            label = f'{config["label"]} {conf:.2f}'
            cv2.rectangle(self.frame, (x1, y1 - 20), (x1 + len(label) * 10, y1), config['color'], -1)
            cv2.putText(self.frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Draw all hoops 
        for hoop in current_frame_hoops:
            x1, y1 = hoop['bbox'][0], hoop['bbox'][1]
            w, h = hoop['size']['width'], hoop['size']['height']
            conf = hoop['confidence']
            
            # Draw the detection box with thicker lines for better visibility
            cvzone.cornerRect(self.frame, (x1, y1, w, h), 
                             colorR=obj_configs['hoop']['color'], 
                             colorC=obj_configs['hoop']['color'],
                             t=3,     # Thickness of the rectangle
                             rt=2)    # Thickness of corner rectangles
            
            # Draw label with background for better readability
            label = f'{obj_configs["hoop"]["label"]} {conf:.2f}'
            cv2.rectangle(self.frame, (x1, y1 - 20), (x1 + len(label) * 10, y1), obj_configs['hoop']['color'], -1)
            cv2.putText(self.frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    def clean_motion(self):
        # Draw ball trajectory with enhanced visibility
        for i in range(0, len(self.ball_pos)):
            cv2.circle(self.frame, self.ball_pos[i][0], 3, (0, 0, 255), -1)  # Filled circles
            # Add a border for better visibility
            cv2.circle(self.frame, self.ball_pos[i][0], 4, (255, 255, 255), 2)

        # Clean hoop motion and display current hoop center
        if len(self.hoop_pos) > 1:
            self.hoop_pos = clean_hoop_pos(self.hoop_pos)
        # Only draw hoop position if we have at least one position
        if len(self.hoop_pos) > 0:
            # Draw hoop position with enhanced visibility
            cv2.circle(self.frame, self.hoop_pos[-1][0], 3, (128, 128, 0), -1)  # Filled circle
            # Add a border for better visibility
            cv2.circle(self.frame, self.hoop_pos[-1][0], 4, (255, 255, 255), 2)

    def shot_detection(self):
        """
        Detect basketball shots based on ball and hoop movement patterns.
        Records shot attempts and determines success based on ball trajectory.
        """
        # Skip if not a detection frame and not at a scene change
        if (self.frame_count % 10 != 0 and 
            (self.next_scene_frame is None or self.frame_count < self.next_scene_frame)):
            return

        # Log basic frame info
        self.logger.debug_log({
            "event": "shot_detection_start",
            "frame": self.frame_count,
            "total_balls": len(self.ball_pos),
            "total_hoops": len(self.hoop_pos),
            "scene_changes_remaining": len(self.scene_changes) if self.scene_changes else 0,
            "next_scene_frame": self.next_scene_frame if self.scene_changes else None
        })

        # Detect UP state (ball approaching hoop from below)
        up = detect_up(self.ball_pos[-1], self.hoop_pos[-1]) if self.ball_pos and self.hoop_pos else False
        self.logger.debug_log({
            "event": "up_detection",
            "frame": self.frame_count,
            "has_ball": bool(self.ball_pos),
            "has_hoop": bool(self.hoop_pos),
            "ball_frame": self.ball_pos[-1][1] if self.ball_pos else None,
            "hoop_frame": self.hoop_pos[-1][1] if self.hoop_pos else None,
            "result": up
        })

        # Update UP state if detection was successful
        if up:
            self.up = True
            self.up_frame = self.ball_pos[-1][1] if self.ball_pos else self.frame_count
            self.up_hoop_frame = self.hoop_pos[-1][1] if self.hoop_pos else self.frame_count
            self.logger.debug_log({
                "event": "up_state_set",
                "frame": self.frame_count,
                "up_frame": self.up_frame,
                "up_hoop_frame": self.up_hoop_frame,
                "ball_confidence": self.ball_pos[-1][4] if self.ball_pos else None,
                "hoop_confidence": self.hoop_pos[-1][4] if self.hoop_pos else None
            })

        # Detect DOWN state (ball passing through hoop area)
        down = False
        if self.up:
            down = detect_down(self.ball_pos[-1], self.hoop_pos[-1]) if self.ball_pos and self.hoop_pos else False
            self.logger.debug_log({
                "event": "down_detection",
                "frame": self.frame_count,
                "has_ball": bool(self.ball_pos),
                "has_hoop": bool(self.hoop_pos),
                "ball_frame": self.ball_pos[-1][1] if self.ball_pos else None,
                "hoop_frame": self.hoop_pos[-1][1] if self.hoop_pos else None,
                "result": down
            })

        # Handle DOWN state
        if self.up and down:
            self.down = True
            self.down_frame = self.ball_pos[-1][1] if self.ball_pos else self.frame_count
            self.down_hoop_frame = self.hoop_pos[-1][1] if self.hoop_pos else self.frame_count
            self.logger.debug_log({
                "event": "down_state_set",
                "frame": self.frame_count,
                "down_frame": self.down_frame,
                "down_hoop_frame": self.down_hoop_frame,
                "ball_confidence": self.ball_pos[-1][4] if self.ball_pos else None,
                "hoop_confidence": self.hoop_pos[-1][4] if self.hoop_pos else None
            })

        # Check if we should perform shot detection
        should_detect_shot = False
        scene_change_triggered = False
        
        # Regular 10-frame interval check
        if self.frame_count % 10 == 0:
            should_detect_shot = True
            self.logger.debug_log({
                "event": "regular_shot_detection",
                "frame": self.frame_count,
                "triggered": True
            })

        # Check for scene change - perform shot detection before the scene change
        if (self.next_scene_frame is not None and self.frame_count >= self.next_scene_frame):
            should_detect_shot = True
            scene_change_triggered = True
            self.logger.debug_log({
                "event": "scene_change_shot_detection",
                "frame": self.frame_count,
                "next_scene_frame": self.next_scene_frame,
                "triggered": True
            })
            
            # Move to next scene change frame
            if self.scene_changes:
                self.scene_changes.pop(0)
                self.next_scene_frame = self.scene_changes[0] if self.scene_changes else None

        # If ball goes from 'up' area to 'down' area in that order, increase attempt and reset
        if self.up and self.down and self.up_frame < self.down_frame and should_detect_shot:
            self.logger.debug_log({
                "event": "shot_conditions_met",
                "frame": self.frame_count,
                "up": self.up,
                "down": self.down,
                "up_frame": self.up_frame,
                "down_frame": self.down_frame,
                "should_detect_shot": should_detect_shot,
                "scene_change_triggered": scene_change_triggered
            })
            
            self.attempts += 1
            self.up = False
            self.down = False

            # Create debug info dictionary
            debug_info = {
                "event": "shot_attempt",
                "frame": self.frame_count,
                "up_frame": self.up_frame,
                "up_hoop_frame": self.up_hoop_frame,
                "down_frame": self.down_frame,
                "down_hoop_frame": self.down_hoop_frame,
                "frames_between_up_down": self.down_frame - self.up_frame,
                "total_ball_positions": len(self.ball_pos),
                "total_hoop_positions": len(self.hoop_pos),
                "scene_change_triggered": scene_change_triggered,
                "ball_positions": [{
                    "frame": pos[1],
                    "position": {"x": pos[0][0], "y": pos[0][1]},
                    "size": {"width": pos[2], "height": pos[3]},
                    "confidence": float(pos[4])
                } for pos in self.ball_pos] if self.ball_pos else [],
                "hoop_positions": [{
                    "frame": pos[1],
                    "position": {"x": pos[0][0], "y": pos[0][1]},
                    "size": {"width": pos[2], "height": pos[3]},
                    "confidence": float(pos[4])
                } for pos in self.hoop_pos] if self.hoop_pos else []
            }
            
            # Check if it's a make or miss with debug info
            is_successful = score(self.ball_pos, self.hoop_pos, debug_info)
            timestamp = self.frame_count / 30  # assuming 30fps
            
            # Log shot (both makes and misses) with debug info
            self.logger.log_shot(
                frame_idx=self.frame_count,
                timestamp=timestamp,
                ball_pos=self.ball_pos[-1][0] if self.ball_pos else (0, 0),
                hoop_pos=self.hoop_pos[-1][0] if self.hoop_pos else (0, 0),
                ball_confidence=self.ball_pos[-1][4] if self.ball_pos else 0.0,
                is_successful=is_successful,
                debug_info=debug_info
            )
            
            # Clear trajectory data to prevent data pollution in subsequent shot detections
            self.ball_pos.clear()
            self.hoop_pos.clear()
            
            if is_successful:
                self.makes += 1
                self.overlay_color = (0, 255, 0)  # Green for make
                self.overlay_text = "Make"
                self.fade_counter = self.fade_frames
                self.logger.debug_log({
                    "event": "successful_shot",
                    "frame": self.frame_count,
                    "make_count": self.makes,
                    "total_attempts": self.attempts
                })
            else:
                self.overlay_color = (255, 0, 0)  # Red for miss
                self.overlay_text = "Miss"
                self.fade_counter = self.fade_frames
                self.logger.debug_log({
                    "event": "missed_shot",
                    "frame": self.frame_count,
                    "miss_count": self.attempts - self.makes,
                    "total_attempts": self.attempts
                })
        
        # If this was triggered by a scene change, reset all tracking data regardless of shot detection
        if scene_change_triggered:
            self.logger.debug_log({
                "event": "scene_change_reset",
                "frame": self.frame_count,
                "scene_change_triggered": scene_change_triggered
            })
            
            # Clear trajectory data to prevent data pollution in subsequent shot detections
            self.ball_pos.clear()
            self.hoop_pos.clear()
            
            # Reset all tracking data for fresh start after scene change
            self.up = False
            self.down = False
            self.up_frame = 0
            self.down_frame = 0
            self.up_hoop_frame = 0
            self.down_hoop_frame = 0
    def display_score(self):
        # Add text with better visibility
        text = str(self.makes) + " / " + str(self.attempts)
        # Draw background rectangle for better contrast
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 3, 6)[0]
        cv2.rectangle(self.frame, (45, 130 - text_size[1]), (45 + text_size[0], 135), (0, 0, 0), -1)
        
        cv2.putText(self.frame, text, (50, 125), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 6)
        cv2.putText(self.frame, text, (50, 125), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 3)

        # Add overlay text for shot result if it exists
        if hasattr(self, 'overlay_text'):
            # Calculate text size to position it at the right top corner
            (text_width, text_height), _ = cv2.getTextSize(self.overlay_text, cv2.FONT_HERSHEY_SIMPLEX, 3, 6)
            text_x = self.frame.shape[1] - text_width - 40  # Right alignment with some margin
            text_y = 100  # Top margin

            # Display overlay text with color (overlay_color) and better visibility
            # Draw background for the overlay text
            cv2.rectangle(self.frame, (text_x - 5, text_y - text_height - 5), 
                         (text_x + text_width + 5, text_y + 5), (0, 0, 0), -1)
            cv2.putText(self.frame, self.overlay_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 3,
                        (255, 255, 255), 6)
            cv2.putText(self.frame, self.overlay_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 3,
                        self.overlay_color, 3)

        # Gradually fade out color after shot
        if self.fade_counter > 0:
            alpha = 0.2 * (self.fade_counter / self.fade_frames)
            self.frame = cv2.addWeighted(self.frame, 1 - alpha, np.full_like(self.frame, self.overlay_color), alpha, 0)
            self.fade_counter -= 1


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Basketball Shot Detector')
    parser.add_argument('--input', type=str, default='video_test_5.mp4', help='Input video file path')
    parser.add_argument('--output', type=str, help='Output video file path')
    parser.add_argument('--model', type=str, default='best.pt', help='Model file path')
    parser.add_argument('--output-dir', type=str, default='output_logs', help='Directory to save logs')
    args = parser.parse_args()
    
    detector = ShotDetector(input_video=args.input, output_video=args.output, model_path=args.model, output_dir=args.output_dir)
    detector.run()