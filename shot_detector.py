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
            video_name = self.input_video.split('/')[-1]
            video_name = video_name.split('.')[0]
            # Generate log filename with video name and start time
            timestamp = self.start_datetime.strftime('%Y%m%d_%H%M%S')
            filename = f"{video_name}_shot_log_{timestamp}.json"
        
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
            video_name = self.input_video.split('/')[-1].split('.')[0]
            timestamp = self.start_datetime.strftime('%Y%m%d_%H%M%S')
            debug_filename = f"{video_name}_frame_log_{timestamp}.json"
            self._debug_file = open(debug_filename, 'w')
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
        Generate detailed debug log file
        
        Args:
            shot_log_path: Path to the original shot log
        """
        # Create debug log filename
        debug_log_path = shot_log_path.replace('_shot_log_', '_debug_log_')
        
        # Extract detailed debug information for each shot
        debug_shots = []
        for i, shot in enumerate(self.shots):
            shot_num = i + 1
            
            # Start with basic shot information
            shot_info = {
                "shot_number": shot_num,
                "frame_index": shot["frame_index"],
                "timestamp": shot["timestamp"],
                "is_successful": shot["is_successful"]
            }
            
            # Add ball position and confidence if available
            if "_ball_position" in shot:
                shot_info["ball_position"] = shot["_ball_position"]
            if "_ball_confidence" in shot:
                shot_info["ball_confidence"] = shot["_ball_confidence"]
            if "_hoop_position" in shot:
                shot_info["hoop_position"] = shot["_hoop_position"]
            
            # Add all debug information
            if "_debug_info" in shot:
                # Convert any non-English strings in debug_info
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
                    shot_info["ball_tracking"] = debug_info["ball_tracking"]
            
                if "hoop_tracking" in debug_info:
                    shot_info["hoop_tracking"] = debug_info["hoop_tracking"]
            
                # Add other debug info
                shot_info["debug_info"] = {k: v for k, v in debug_info.items() 
                                          if k not in ["ball_tracking", "hoop_tracking"]}
                
                # Add concise result reason
                if shot["is_successful"] and "success_reason" in debug_info:
                    shot_info["result_reason"] = debug_info["success_reason"]
                elif not shot["is_successful"] and "failure_reason" in debug_info:
                    shot_info["result_reason"] = debug_info["failure_reason"]
            
            debug_shots.append(shot_info)
        
        # Create debug log data
        debug_data = {
            "video": self.input_video,
            "total_shots": self.total_attempts,
            "successful_shots": self.success_count,
            "shooting_accuracy": float(self.success_count / self.total_attempts * 100) if self.total_attempts > 0 else 0.0,
            "detailed_shots": debug_shots
        }
        
        # Save debug log
        with open(debug_log_path, 'w') as f:
            json.dump(debug_data, f, indent=2)
        
        print(f"Detailed debug log saved to {debug_log_path}")
        return debug_log_path

class ShotDetector:
    def __init__(self, input_video="video_test_5.mp4", output_video=None, ball_model_path="yolov8m.pt", hoop_model_path="best.pt"):
        # Load dual models for optimal detection
        self.overlay_text = "Waiting..."

        # Load ball detection model (YOLOv8m for sports ball)
        self.ball_model_path = ball_model_path
        self.ball_model = YOLO(ball_model_path)
        print(f"🏀 Loaded ball model: {ball_model_path}")

        # Load hoop detection model (custom trained model)
        self.hoop_model_path = hoop_model_path
        self.hoop_model = YOLO(hoop_model_path)
        print(f"🏀 Loaded hoop model: {hoop_model_path}")

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
            print(f"📋 Hoop model classes: {list(self.hoop_model_classes.values())}")
        else:
            self.hoop_model_classes = {0: "Basketball", 1: "Basketball Hoop"}
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

            # Run dual model detection
            ball_results = self.ball_model(self.frame, stream=True, device=self.device, verbose=False)
            hoop_results = self.hoop_model(self.frame, stream=True, device=self.device, verbose=False)

            # Collect all detections in current frame for logging
            current_frame_balls = []
            current_frame_hoops = []

            # Process ball detections from YOLOv8m (sports ball)
            for r in ball_results:
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

                        if is_ball:
                            current_frame_balls.append({
                                "bbox": [x1, y1, x2, y2],
                                "center": center,
                                "size": {"width": w, "height": h},
                                "confidence": float(conf),
                                "class": current_class
                            })

                            # Only create ball points if reasonable confidence or near hoop
                            if (conf > 0.3 or (in_hoop_region(center, self.hoop_pos) and conf > 0.15)):
                                self.ball_pos.append((center, self.frame_count, w, h, conf))
                                cvzone.cornerRect(self.frame, (x1, y1, w, h))

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
                                  "hoop" in current_class.lower() or "basket" in current_class.lower())

                        if is_hoop:
                            current_frame_hoops.append({
                                "bbox": [x1, y1, x2, y2],
                                "center": center,
                                "size": {"width": w, "height": h},
                                "confidence": float(conf),
                                "class": current_class
                            })

                            # Create hoop points if high confidence
                            if conf > 0.5:
                                self.hoop_pos.append((center, self.frame_count, w, h, conf))
                                cvzone.cornerRect(self.frame, (x1, y1, w, h))

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
        
        # Close frame log file if it exists and is still open
        if hasattr(self.logger, '_debug_file') and self.logger._debug_file and not self.logger._debug_file.closed:
            try:
                self.logger._debug_file.write(']')  # Close JSON array
                self.logger._debug_file.close()
            except (ValueError, AttributeError) as e:
                print(f"Warning: Could not close debug file properly: {e}")
        
        # Save shot log after processing completes
        log_filename = self.logger.save_log()
        print(f"\nShot log saved to: {log_filename}")

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
    parser.add_argument('--ball-model', type=str, default='yolov8m.pt', help='Ball detection model (default: yolov8m.pt)')
    parser.add_argument('--hoop-model', type=str, default='best.pt', help='Hoop detection model (default: best.pt)')
    args = parser.parse_args()

    # Create detector with dual models
    detector = ShotDetector(args.input, args.output, args.ball_model, args.hoop_model)
    detector.run()
    ShotDetector(input_video=args.input, output_video=args.output)