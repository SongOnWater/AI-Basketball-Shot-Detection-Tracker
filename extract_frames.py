
import cv2
import os
import argparse

def extract_frames(video_path, output_dir, frames=None, frame_range=None):

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # --- 自动生成输出目录名 ---
    if output_dir is None:
        base = os.path.splitext(os.path.basename(video_path))[0]
        output_dir = base
        orig_output_dir = output_dir
        count = 1
        while os.path.exists(output_dir):
            output_dir = f"{orig_output_dir}_{count}"
            count += 1
    os.makedirs(output_dir, exist_ok=True)

    # Determine which frames to extract
    if frames is not None:
        frame_indices = sorted(set(frames))
    elif frame_range is not None:
        start, end = frame_range
        frame_indices = list(range(start, min(end, total_frames)))
    else:
        frame_indices = list(range(total_frames))

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            print(f"Warning: Could not read frame {idx}")
            continue
        out_path = os.path.join(output_dir, f"frame_{idx:05d}.jpg")
        cv2.imwrite(out_path, frame)
        print(f"Saved {out_path}")

    cap.release()
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames from a video file.")
    parser.add_argument("video", help="Path to the video file.")
    parser.add_argument("--output", default=None, help="Directory to save extracted frames. If not set, will use video name.")
    parser.add_argument("--frames", nargs="*", type=int, default=None, help="Specific frame indices to extract (space separated)")
    parser.add_argument("--range", nargs=2, type=int, default=None, metavar=("START", "END"), help="Frame range to extract (inclusive start, exclusive end)")
    args = parser.parse_args()

    extract_frames(
        args.video,
        args.output,
        frames=args.frames if args.frames else None,
        frame_range=args.range if args.range else None
    )
