#!/usr/bin/env python3
import os
import json
import argparse
from collections import defaultdict
from shot_detector import ShotDetector
import tempfile
import shutil
import glob
from datetime import datetime

def load_ground_truth(txt_file):
    """
    Load ground truth data from txt file
    Each line contains: frame_number label (1 for success, 0 for miss)
    """
    ground_truth = []
    with open(txt_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split()
                if len(parts) == 2:
                    frame_number = int(parts[0])
                    label = int(parts[1])
                    ground_truth.append({
                        'frame': frame_number,
                        'successful': bool(label)
                    })
    return ground_truth

def evaluate_shots(ground_truth, detected_shots, frame_tolerance=30):
    """
    Evaluate detected shots against ground truth
    
    Args:
        ground_truth: List of ground truth shots with frame numbers
        detected_shots: List of detected shots from shot detector
        frame_tolerance: Number of frames after ground truth frame to consider a match
    
    Returns:
        Dictionary with evaluation metrics
    """
    # Match detected shots to ground truth
    tp, fp, fn = 0, 0, 0  # True positives, false positives, false negatives
    tp_details = []
    fp_details = []
    fn_details = []
    
    # Create a copy of ground truth for tracking unmatched items
    unmatched_gt = ground_truth.copy()
    
    # For each detected shot, try to match it with a ground truth shot
    for detected_shot in detected_shots:
        detected_frame = detected_shot['frame_index']
        detected_successful = detected_shot['is_successful']
        
        matched = False
        # Check if this detected shot matches any ground truth shot
        for i, gt_shot in enumerate(unmatched_gt):
            gt_frame = gt_shot['frame']
            gt_successful = gt_shot['successful']
            
            # Check if detected shot is within tolerance window after ground truth frame
            if gt_frame <= detected_frame <= gt_frame + frame_tolerance:
                # Check if success/failure classification matches
                if detected_successful == gt_successful:
                    # Match found
                    tp += 1
                    tp_details.append({
                        'ground_truth_frame': gt_frame,
                        'detected_frame': detected_frame,
                        'ground_truth_success': gt_successful,
                        'detected_success': detected_successful,
                        'frame_diff': detected_frame - gt_frame
                    })
                    # Remove matched ground truth
                    unmatched_gt.pop(i)
                    matched = True
                    break
                else:
                    # Detected but with wrong classification
                    tp += 1  # Still count as detected
                    tp_details.append({
                        'ground_truth_frame': gt_frame,
                        'detected_frame': detected_frame,
                        'ground_truth_success': gt_successful,
                        'detected_success': detected_successful,
                        'frame_diff': detected_frame - gt_frame,
                        'classification_error': True
                    })
                    # Remove matched ground truth
                    unmatched_gt.pop(i)
                    matched = True
                    break
        
        if not matched:
            # False positive - detected shot with no corresponding ground truth
            fp += 1
            fp_details.append({
                'detected_frame': detected_frame,
                'detected_success': detected_successful
            })
    
    # Remaining unmatched ground truth shots are false negatives
    for gt_shot in unmatched_gt:
        fn += 1
        fn_details.append({
            'ground_truth_frame': gt_shot['frame'],
            'ground_truth_success': gt_shot['successful']
        })
    
    # Calculate metrics
    total_gt_shots = len(ground_truth)
    total_detected_shots = len(detected_shots)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'total_ground_truth_shots': total_gt_shots,
        'total_detected_shots': total_detected_shots,
        'details': {
            'true_positives': tp_details,
            'false_positives': fp_details,
            'false_negatives': fn_details
        }
    }

def find_shot_log(video_name):
    """Find the shot log file for a given video name"""
    # Look for files in the current directory
    import glob
    import os
    
    # Pattern to match shot log files with timestamps
    patterns = [
        f"{video_name}*_shot_????-??-??_??-??-??.json",  # With model name
        f"{video_name}_shot_????-??-??_??-??-??.json",   # Without model name
        f"{video_name}*_shot_log_????????_??????.json",  # Old format with model name
        f"{video_name}_shot_log_????????_??????.json"    # Old format without model name
    ]
    
    # Look in current directory first
    for pattern in patterns:
        matches = glob.glob(pattern)
        if matches:
            # Return the most recent file (sort by modification time)
            return max(matches, key=os.path.getmtime)
    
    # Look in logs directory
    if os.path.exists('logs'):
        for pattern in patterns:
            matches = glob.glob(os.path.join('logs', pattern))
            if matches:
                # Return the most recent file (sort by modification time)
                return max(matches, key=os.path.getmtime)
    
    return None

def move_log_files(shot_log_file, output_dir):
    """Move log files to output directory"""
    if not output_dir:
        return
        
    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)
    
    # Move shot log
    final_shot_log = os.path.join(output_dir, os.path.basename(shot_log_file))
    if shot_log_file != final_shot_log:
        shutil.move(shot_log_file, final_shot_log)
        shot_log_file = final_shot_log
    
    # Move frame log if it exists
    base_name = os.path.splitext(os.path.basename(shot_log_file))[0]
    
    # Try to find frame log with new naming convention
    frame_log_candidates = [
        f"{base_name.replace('_shot_', '_frame_')}.json",
        f"{base_name.replace('_shot_', '_frame_')}.txt",  # For backward compatibility
        f"{base_name}_frame.json",
        f"{base_name}_frame.txt"  # For backward compatibility
    ]
    
    # Also try old naming conventions
    if "_shot_" in base_name:
        prefix = base_name.split("_shot_")[0]
        frame_log_candidates.extend([
            f"{prefix}_frame*.json",
            f"{prefix}_frame*.txt"
        ])
    
    frame_log_file = None
    for candidate in frame_log_candidates:
        # Handle glob patterns
        if "*" in candidate:
            matches = glob.glob(candidate)
            if matches:
                frame_log_file = matches[0]  # Take the first match
                break
        else:
            if os.path.exists(candidate):
                frame_log_file = candidate
                break
    
    if frame_log_file:
        final_frame_log = os.path.join(output_dir, os.path.basename(frame_log_file))
        if frame_log_file != final_frame_log:
            shutil.move(frame_log_file, final_frame_log)

def process_video(video_path, txt_path, output_dir=None, model_path=None):
    """
    Process a single video and evaluate results
    
    Args:
        video_path: Path to the input video file
        txt_path: Path to the ground truth txt file
        output_dir: Directory to store output files (logs, videos)
        model_path: Path to the model file to use for detection
    """
    print(f"Processing video: {video_path}")
    
    # Create output directory if specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Load ground truth
    ground_truth = load_ground_truth(txt_path)
    print(f"Loaded {len(ground_truth)} ground truth shots from {txt_path}")
    
    # Create temporary directory for log files
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Determine output video path
        output_video_path = None
        if output_dir:
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            output_video_path = os.path.join(output_dir, f"{video_name}_output.mp4")
        
        # Determine debug log path
        debug_log_path = None
        if output_dir:
            model_name = os.path.splitext(os.path.basename(model_path))[0] if model_path else "default"
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            debug_log_path = os.path.join(output_dir, f"{video_name}_{model_name}_debug_{timestamp}.txt")
        
        # Process video with shot detector
        detector = ShotDetector(
            input_video=video_path, 
            output_video=output_video_path, 
            ball_model_path=model_path,  # This is compatible with our modified ShotDetector
            output_dir=output_dir,  # Pass output directory to ShotDetector
            debug_log_path=debug_log_path  # 传递调试日志路径
        )
        
        # Actually run the detector to process the video
        print("Running shot detection...")
        shot_log_file = detector.run()  # Get the shot log file path directly from the detector
        print("Shot detection completed.")
        
        # Find the shot log file (it should be in logs directory)
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        model_name = os.path.splitext(os.path.basename(model_path))[0] if model_path else "default"
        
        # Use the returned shot log file path directly
        if not shot_log_file or not os.path.exists(shot_log_file):
            print(f"No shot log file found for {video_name}")
            return None
            
        print(f"Loading shot log: {shot_log_file}")
        
        with open(shot_log_file, 'r') as f:
            shot_log_data = json.load(f)
        
        detected_shots = shot_log_data.get('shots', [])
        # Extract shot data from entries if needed
        if detected_shots and 'shot' in detected_shots[0]:
            detected_shots = [entry['shot'] for entry in detected_shots]
            
        print(f"Detected {len(detected_shots)} shots")
        
        # Evaluate
        evaluation = evaluate_shots(ground_truth, detected_shots)
        evaluation['video_name'] = video_name
        
        # Move log files to output directory if specified
        move_log_files(shot_log_file, output_dir)
        
        evaluation['shot_log_file'] = shot_log_file if os.path.exists(shot_log_file) else None
        return evaluation
        
    except Exception as e:
        print(f"Error processing video {video_path}: {e}")
        traceback.print_exc()
        return None
    
    finally:
        # Clean up temporary directory
        try:
            shutil.rmtree(temp_dir)
        except:
            pass

def generate_report(results):
    """
    Generate a comprehensive evaluation report
    """
    if not results:
        print("No results to report")
        return
    
    print("\n" + "="*80)
    print("BASKETBALL SHOT DETECTION EVALUATION REPORT")
    print("="*80)
    
    total_tp = sum(r['tp'] for r in results)
    total_fp = sum(r['fp'] for r in results)
    total_fn = sum(r['fn'] for r in results)
    total_gt_shots = sum(r['total_ground_truth_shots'] for r in results)
    total_detected_shots = sum(r['total_detected_shots'] for r in results)
    
    overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0
    
    print(f"\nOVERALL RESULTS:")
    print(f"  Total Ground Truth Shots: {total_gt_shots}")
    print(f"  Total Detected Shots: {total_detected_shots}")
    print(f"  True Positives: {total_tp}")
    print(f"  False Positives: {total_fp}")
    print(f"  False Negatives: {total_fn}")
    print(f"  Precision: {overall_precision:.4f}")
    print(f"  Recall: {overall_recall:.4f}")
    print(f"  F1-Score: {overall_f1:.4f}")
    
    print(f"\nPER-VIDEO RESULTS:")
    for result in results:
        print(f"\n  Video: {result['video_name']}")
        print(f"    Ground Truth Shots: {result['total_ground_truth_shots']}")
        print(f"    Detected Shots: {result['total_detected_shots']}")
        print(f"    True Positives: {result['tp']}")
        print(f"    False Positives: {result['fp']}")
        print(f"    False Negatives: {result['fn']}")
        print(f"    Precision: {result['precision']:.4f}")
        print(f"    Recall: {result['recall']:.4f}")
        print(f"    F1-Score: {result['f1_score']:.4f}")
    
    print("\n" + "="*80)
    print("DETAILED ANALYSIS")
    print("="*80)
    
    for result in results:
        print(f"\nVideo: {result['video_name']}")
        details = result['details']
        
        if details['true_positives']:
            print(f"  True Positives ({len(details['true_positives'])}):")
            for tp in details['true_positives'][:5]:  # Show first 5
                if tp.get('classification_error'):
                    print(f"    Frame {tp['ground_truth_frame']} -> {tp['detected_frame']} "
                          f"(Diff: {tp['frame_diff']} frames) - CLASSIFICATION ERROR")
                else:
                    print(f"    Frame {tp['ground_truth_frame']} -> {tp['detected_frame']} "
                          f"(Diff: {tp['frame_diff']} frames)")
            if len(details['true_positives']) > 5:
                print(f"    ... and {len(details['true_positives']) - 5} more")
        
        if details['false_positives']:
            print(f"  False Positives ({len(details['false_positives'])}):")
            for fp in details['false_positives'][:5]:  # Show first 5
                print(f"    Detected at frame {fp['detected_frame']} (Success: {fp['detected_success']})")
            if len(details['false_positives']) > 5:
                print(f"    ... and {len(details['false_positives']) - 5} more")
        
        if details['false_negatives']:
            print(f"  False Negatives ({len(details['false_negatives'])}):")
            for fn in details['false_negatives'][:5]:  # Show first 5
                print(f"    Ground truth at frame {fn['ground_truth_frame']} (Success: {fn['ground_truth_success']})")
            if len(details['false_negatives']) > 5:
                print(f"    ... and {len(details['false_negatives']) - 5} more")

def main(folder_path, frame_tolerance=30, output_dir=None, model_path=None, input_indices=None):
    """
    Main function to process all video-txt pairs in a folder
    
    Args:
        folder_path: Path to folder containing video-txt pairs
        frame_tolerance: Frame tolerance for matching detected shots to ground truth
        output_dir: Directory to store output files (logs, videos)
        model_path: Path to the model file to use for detection
        input_indices: List of indices to process. If None, process all files.
                      If [-1], process all files. Otherwise, process only specified indices.
    """
    print(f"Starting batch evaluation on folder: {folder_path}")
    print(f"Frame tolerance: {frame_tolerance} frames")
    if model_path:
        print(f"Model path: {model_path}")
    if output_dir:
        print(f"Output directory: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
    
    # Check if folder exists
    if not os.path.exists(folder_path):
        print(f"Error: Folder {folder_path} does not exist")
        return
    
    # Find all video-txt pairs
    video_files = [f for f in os.listdir(folder_path) if f.endswith('.mp4')]
    
    if not video_files:
        print(f"No video files found in {folder_path}")
        return
    
    # Sort video files for consistent indexing
    video_files.sort()
    
    # Handle input indices
    if input_indices is None:
        # List all videos with indices and prompt user to select
        print("\nAvailable videos:")
        for i, video_file in enumerate(video_files):
            print(f"  [{i}]: {video_file}")
        
        print("\nEnter indices to process (comma-separated), or -1 for all videos:")
        try:
            user_input = input().strip()
            if user_input == "-1":
                selected_indices = [-1]
            else:
                selected_indices = [int(x.strip()) for x in user_input.split(",") if x.strip()]
        except ValueError:
            print("Invalid input. Please enter comma-separated integers or -1.")
            return
    else:
        selected_indices = input_indices
    
    # Filter video files based on selected indices
    if selected_indices == [-1] or selected_indices is None:
        # Process all videos
        selected_video_files = video_files
        print(f"Processing all {len(selected_video_files)} videos")
    else:
        # Process only selected indices
        selected_video_files = []
        for idx in selected_indices:
            if 0 <= idx < len(video_files):
                selected_video_files.append(video_files[idx])
            else:
                print(f"Warning: Index {idx} is out of range (0-{len(video_files)-1}). Skipping.")
        
        print(f"Processing {len(selected_video_files)} selected videos")
    
    if not selected_video_files:
        print("No valid videos selected for processing.")
        return
    
    results = []
    
    for video_file in selected_video_files:
        video_path = os.path.join(folder_path, video_file)
        txt_file = os.path.splitext(video_path)[0] + '.txt'
        
        # Check if corresponding txt file exists
        if not os.path.exists(txt_file):
            print(f"Warning: No corresponding txt file for {video_file}")
            continue
        
        # Process the video
        result = process_video(video_path, txt_file, output_dir, model_path)
        if result:
            results.append(result)
    
    # Generate report
    generate_report(results)
    
    # Save report to file if output directory is specified
    if output_dir:
        report_path = os.path.join(output_dir, "batch_test_report.txt")
        with open(report_path, 'w') as f:
            # Redirect print output to file
            import sys
            original_stdout = sys.stdout
            sys.stdout = f
            generate_report(results)
            sys.stdout = original_stdout
        print(f"Report saved to: {report_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Batch test basketball shot detector')
    parser.add_argument('--folder', type=str, default='video_txt_pairs', 
                        help='Folder containing video-txt pairs')
    parser.add_argument('--tolerance', type=int, default=30,
                        help='Frame tolerance for matching detected shots to ground truth (default: 30)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Directory to store output files (logs, videos). If not specified, files are stored in their default locations and cleaned up.')
    parser.add_argument('--model', type=str, default='best.pt',
                        help='Path to the model file to use for detection (default: best.pt)')
    parser.add_argument('--input-index', type=str, default=None,
                        help='Comma-separated indices of videos to process. Use -1 for all videos. If not specified, user will be prompted.')
    
    args = parser.parse_args()
    
    # Parse input indices
    input_indices = None
    if args.input_index is not None:
        if args.input_index.strip() == "-1":
            input_indices = [-1]
        else:
            try:
                input_indices = [int(x.strip()) for x in args.input_index.split(",") if x.strip()]
            except ValueError:
                print("Error: --input-index must be comma-separated integers or -1")
                exit(1)
    
    main(args.folder, args.tolerance, args.output_dir, args.model, input_indices)