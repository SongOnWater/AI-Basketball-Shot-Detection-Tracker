
import cv2
import os
import argparse
import subprocess
import shutil
import sys

def check_ffmpeg():
    """检查系统中是否安装了ffmpeg"""
    try:
        result = subprocess.run(['ffmpeg', '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False

def extract_frames_with_ffmpeg(video_path, output_dir, frames=None, frame_range=None):
    """使用ffmpeg抽取视频帧"""
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 构建ffmpeg命令
    if frames is not None:
        # 如果指定了特定帧，我们需要为每一帧单独运行ffmpeg
        for idx in sorted(set(frames)):
            vidx = idx + 1  # 调整为1-based索引以保持全局帧编号
            out_path = os.path.join(output_dir, f"frame_{vidx:05d}.jpg")
            # ffmpeg -i video.mp4 -vf "select=eq(n\,10)" -vframes 1 output.jpg
            cmd = [
                'ffmpeg', '-i', video_path, 
                '-vf', f'select=eq(n\\,{idx})', 
                '-vframes', '1', 
                '-q:v', '1',  # 高质量
                out_path
            ]
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print(f"Saved {out_path}")
    else:
        # 如果是提取所有帧或一个范围内的帧
        if frame_range is not None:
            start, end = frame_range
            # 使用ffmpeg的start_number选项来保持全局帧编号
            # 这样可以避免循环，提高效率
            out_pattern = os.path.join(output_dir, f"frame_%05d.jpg")
            cmd = [
                'ffmpeg', '-i', video_path,
                '-vf', f'select=between(n\\,{start}\\,{end-1})',  # 选择范围内的帧
                '-vsync', '0',
                '-q:v', '1',  # 高质量
                '-start_number', str(start + 1),  # 设置起始编号以保持全局帧编号
                out_pattern
            ]
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            # 打印保存的帧信息
            for idx in range(start, end):
                vidx = idx + 1
                out_path = os.path.join(output_dir, f"frame_{vidx:05d}.jpg")
                print(f"Saved {out_path}")
        else:
            cmd = [
                'ffmpeg', '-i', video_path,
                '-vf', 'select=1',
                '-vsync', '0',
                '-q:v', '1',  # 高质量
                os.path.join(output_dir, f"frame_%05d.jpg")
            ]
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print(f"Extracted frames saved to {output_dir}")
    
    print("Done.")

def extract_frames(video_path, output_dir, frames=None, frame_range=None):
    """
    从视频中提取帧。如果系统中有ffmpeg，则使用ffmpeg；否则使用OpenCV。
    
    Args:
        video_path: 视频文件路径
        output_dir: 输出目录路径
        frames: 要提取的特定帧索引列表
        frame_range: 要提取的帧范围 (start, end)
    """
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
    
    # 检查是否有ffmpeg
    has_ffmpeg = check_ffmpeg()
    print(f"FFmpeg available: {has_ffmpeg}")  # 添加打印以验证ffmpeg可用性
    if has_ffmpeg:
        print("检测到ffmpeg，使用ffmpeg提取帧...")
        extract_frames_with_ffmpeg(video_path, output_dir, frames, frame_range)
        print("使用ffmpeg完成帧提取")  # 添加完成打印
    else:
        print("未检测到ffmpeg，使用OpenCV提取帧...")
        extract_frames_with_opencv(video_path, output_dir, frames, frame_range)
        print("使用OpenCV完成帧提取")  # 添加完成打印

def extract_frames_with_opencv(video_path, output_dir, frames=None, frame_range=None):
    """使用OpenCV提取视频帧"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

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
        vidx=idx+1  # 保持全局帧编号（从1开始）
        if not ret:
            print(f"Warning: Could not read frame {vidx}")
            continue
        out_path = os.path.join(output_dir, f"frame_{vidx:05d}.jpg")
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
