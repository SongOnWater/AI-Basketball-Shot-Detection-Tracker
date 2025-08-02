# Basketball Shot Detection Batch Testing

This script allows you to evaluate the performance of the basketball shot detector against ground truth data.

## Data Format

The ground truth data is stored in pairs of video files (`.mp4`) and text files (`.txt`) with the same base name.

In each text file:
- Each line contains a frame number and a label (1 for successful shot, 0 for missed shot)
- The frame number represents the exact frame when the ball passes through the hoop

## Usage

To run the batch test:

```bash
python batch_test_evaluator.py [--folder FOLDER] [--tolerance TOLERANCE] [--output-dir OUTPUT_DIR]
```

### Parameters

- `--folder`: Folder containing video-txt pairs (default: `video_txt_pairs`)
- `--tolerance`: Frame tolerance for matching detected shots to ground truth (default: 30 frames)
- `--output-dir`: Directory to store output files (logs, videos). If not specified, files are stored in their default locations and cleaned up.

### Example

```bash
# Run without saving output files
python batch_test_evaluator.py --folder video_txt_pairs --tolerance 30

# Run and save all output files to a specific directory
python batch_test_evaluator.py --folder video_txt_pairs --tolerance 30 --output-dir batch_test_results
```

## How It Works

1. The script processes each video file in the specified folder using the shot detector
2. It compares the detected shots with the ground truth data from the corresponding txt file
3. A detected shot is considered a match if:
   - It occurs within the tolerance window (default 30 frames) after a ground truth frame
   - The success/failure classification matches the ground truth

## Evaluation Metrics

The script calculates the following metrics:

- **True Positives (TP)**: Correctly detected shots
- **False Positives (FP)**: Detected shots with no corresponding ground truth
- **False Negatives (FN)**: Ground truth shots that were not detected
- **Precision**: TP / (TP + FP)
- **Recall**: TP / (TP + FN)
- **F1-Score**: Harmonic mean of precision and recall

## Output Files

When using the `--output-dir` option, the following files are saved to the specified directory:

1. **Shot Detection Logs**: JSON files containing shot detection results (`[video_name]_shot_log_[timestamp].json`)
2. **Debug Logs**: Detailed debug information (`[video_name]_debug_log_[timestamp].json`)
3. **Frame Logs**: Frame-by-frame processing data (`[video_name]_frame_log_[timestamp].json`)
4. **Output Videos**: Processed videos with annotations (`[video_name]_output.mp4`)
5. **Test Report**: Summary of batch test results (`batch_test_report.txt`)

## Output

The script generates a comprehensive report including:
- Overall performance metrics
- Per-video performance metrics
- Detailed analysis of matches, false positives, and false negatives