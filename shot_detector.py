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
        
    def set_output_info(self, output_dir, model_name):
        """Set output directory and model name for consistent log naming"""
        self.output_dir = output_dir
        self.model_name = model_name
        
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
            "shots": clean_shots
        }
        with open(filename, 'w') as f:
            json.dump(stats, f, indent=2)
        
        # Generate detailed debug log
        self.save_debug_log(filename)
        
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
                frame_filename = f"{video_name}_frame_{timestamp}.json"
                
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
            debug_log_path = f"{parts[0]}.json"
        else:
            # Fallback: just remove "_shot" if present
            debug_log_path = base_name.replace("_shot", "") + ".json"
        
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

class ShotDetector:
    def __init__(self, input_video="video_test_5.mp4", output_video=None, model_path="best.pt", 
                 ball_model_path=None, hoop_model_path=None, person_model_path=None, use_shared_model=True, 
                 min_ball_area=400, enable_person_detection=False, model_config=None, debug_log_path=None, 
                 output_dir=None):
        # For compatibility with batch_test_evaluator.py
        # Use ball_model_path if provided, otherwise fall back to model_path
        actual_model_path = ball_model_path if ball_model_path else model_path
        
        # Load the YOLO model created from main.py - change text to your relative path
        self.overlay_text = "Waiting..."
        self.model = YOLO(actual_model_path)
        self.model_path = actual_model_path
        self.output_video = output_video
        self.video_writer = None
        self.input_video = input_video
        self.output_dir = output_dir  # Store output directory
        
        # Set up debug log path
        self.debug_log_path = debug_log_path or os.path.join('logs', 'console_output.log')
        os.makedirs(os.path.dirname(self.debug_log_path), exist_ok=True)
        
        # Set up class names based on the model
        self.setup_class_names()
        
        # Create logger after setting up debug_log_path
        self.logger = ShotLogger(input_video=self.input_video)
        # Set output directory and model name for consistent log naming
        model_name = os.path.splitext(os.path.basename(self.model_path))[0] if self.model_path else "default"
        self.logger.set_output_info(self.output_dir, model_name)
        
        # Uncomment this line to accelerate inference. Note that this may cause errors in some setups.
        #self.model.half()
        
        self.device = get_device()
        # Uncomment line below to use webcam (I streamed to my iPhone using Iriun Webcam)
        # self.cap = cv2.VideoCapture(0)

        # Use video from input parameter
        self.cap = cv2.VideoCapture(input_video)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

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

        # Used for green and red colors after make/miss
        self.fade_frames = 20
        self.fade_counter = 0
        self.overlay_color = (0, 0, 0)

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
            
    def run(self):
        # Progress bar
        progress_bar = tqdm(total=self.total_frames, desc="Processing Video", unit="frame")
        
        # Create output directory if not exists
        if self.logger.output_dir:
            os.makedirs(self.logger.output_dir, exist_ok=True)
            
        # Create frame log file with consistent naming and .json extension
        video_name = os.path.splitext(os.path.basename(self.input_video))[0]
        timestamp = self.logger.start_datetime.strftime('%Y-%m-%d_%H-%M-%S')
        
        # Generate consistent naming for frame log with model name and .json extension
        if self.logger.model_name:
            frame_log_filename = f"{video_name}_{self.logger.model_name}_frame_{timestamp}.json"
        else:
            frame_log_filename = f"{video_name}_frame_log_{timestamp}.json"
            
        frame_log_path = os.path.join(self.logger.output_dir or '.', frame_log_filename)
        self.logger._debug_file = open(frame_log_path, 'w')
        self.logger._debug_file.write('[\n')  # Start JSON array

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
                        'class': class_name
                    }

                    # Classify detected object
                    if class_name.lower() in ['ball', 'sports ball']:
                        # Ball detection
                        current_frame_balls.append(detection_data)
                    elif class_name.lower() in ['hoop', 'basketball hoop', 'rim']:
                        # Hoop detection
                        current_frame_hoops.append(detection_data)

                    # Draw bounding boxes
                    if conf > 0.5:  # Only draw high confidence detections
                        if class_name.lower() in ['ball', 'sports ball']:
                            cvzone.cornerRect(self.frame, (x1, y1, w, h), colorR=(0, 0, 255), colorC=(0, 0, 255))  # Red for ball
                            cv2.putText(self.frame, f'Ball {conf}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        elif class_name.lower() in ['hoop', 'basketball hoop', 'rim']:
                            cvzone.cornerRect(self.frame, (x1, y1, w, h), colorR=(255, 0, 0), colorC=(255, 0, 0))  # Blue for hoop
                            cv2.putText(self.frame, f'Hoop {conf}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # Add ball and hoop positions to tracking arrays
            for ball in current_frame_balls:
                # Create ball points if high confidence
                if ball['confidence'] > 0.5:
                    center = ball['center']
                    w = ball['size']['width']
                    h = ball['size']['height']
                    conf = ball['confidence']
                    self.ball_pos.append((center, self.frame_count, w, h, conf))

            for hoop in current_frame_hoops:
                # Create hoop points if high confidence
                if hoop['confidence'] > 0.5:
                    center = hoop['center']
                    w = hoop['size']['width']
                    h = hoop['size']['height']
                    conf = hoop['confidence']
                    self.hoop_pos.append((center, self.frame_count, w, h, conf))
                    cvzone.cornerRect(self.frame, (hoop['bbox'][0], hoop['bbox'][1], w, h))

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
        
        # Save shot log after processing completes
        log_filename = self.logger.save_log()
        print(f"\nShot log saved to: {log_filename}")
        
        return log_filename  # Return log filename for batch evaluator

    def clean_motion(self):
        # Clean and display ball motion
        self.ball_pos = clean_ball_pos(self.ball_pos, self.frame_count)
        for i in range(0, len(self.ball_pos)):
            cv2.circle(self.frame, self.ball_pos[i][0], 2, (0, 0, 255), 2)

        # Clean hoop motion and display current hoop center
        if len(self.hoop_pos) > 1:
            self.hoop_pos = clean_hoop_pos(self.hoop_pos)
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

            # If ball goes from 'up' area to 'down' area in that order, increase attempt and reset
            if self.frame_count % 10 == 0:
                if self.up and self.down and self.up_frame < self.down_frame:
                    self.attempts += 1
                    self.up = False
                    self.down = False

                    # Create debug info dictionary
                    debug_info = {}
                    
                    # Add more context information to debug dictionary
                    debug_info['shot_context'] = {
                        'up_frame': self.up_frame,
                        'down_frame': self.down_frame,
                        'frames_between_up_down': self.down_frame - self.up_frame,
                        'total_ball_positions': len(self.ball_pos),
                        'total_hoop_positions': len(self.hoop_pos)
                    }
                    
                    # Add detailed ball and hoop tracking data for each frame
                    ball_tracking_data = []
                    for pos in self.ball_pos:
                        ball_tracking_data.append({
                            'frame': pos[1],
                            'position': {'x': pos[0][0], 'y': pos[0][1]},
                            'size': {'width': pos[2], 'height': pos[3]},
                            'confidence': float(pos[4])
                        })
                    
                    hoop_tracking_data = []
                    for pos in self.hoop_pos:
                        hoop_tracking_data.append({
                            'frame': pos[1],
                            'position': {'x': pos[0][0], 'y': pos[0][1]},
                            'size': {'width': pos[2], 'height': pos[3]},
                            'confidence': float(pos[4])
                        })
                    
                    debug_info['ball_tracking'] = ball_tracking_data
                    debug_info['hoop_tracking'] = hoop_tracking_data
                    
                    # Check if it's a make or miss with debug info
                    is_successful = score(self.ball_pos, self.hoop_pos, debug_info)
                    timestamp = self.frame_count / 30  # assuming 30fps
                    
                    # Log shot (both makes and misses) with debug info
                    self.logger.log_shot(
                        frame_idx=self.frame_count,
                        timestamp=timestamp,
                        ball_pos=self.ball_pos[-1][0],
                        hoop_pos=self.hoop_pos[-1][0],
                        ball_confidence=self.ball_pos[-1][4],  # Use actual ball confidence
                        is_successful=is_successful,
                        debug_info=debug_info
                    )
                    
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
    parser = argparse.ArgumentParser(description='Basketball Shot Detector')
    parser.add_argument('--input', type=str, default='video_test_5.mp4', help='Input video file path')
    parser.add_argument('--output', type=str, help='Output video file path')
    parser.add_argument('--model', type=str, default='best.pt', help='Model file path')
    parser.add_argument('--output-dir', type=str, default='output_logs', help='Directory to save logs')
    args = parser.parse_args()
    
    detector = ShotDetector(input_video=args.input, output_video=args.output, model_path=args.model, output_dir=args.output_dir)
    detector.run()