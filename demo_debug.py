#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
篮球投篮检测系统调试功能演示脚本
"""

import os
import argparse
import subprocess
import glob
import time
from shot_detector import ShotDetector
from analyze_debug_logs import main as analyze_logs

def run_shot_detection(input_video, output_video):
    """
    运行投篮检测系统
    
    Args:
        input_video: 输入视频路径
        output_video: 输出视频路径
    
    Returns:
        str: 生成的调试日志文件路径
    """
    print("\n===== 步骤 1: 运行投篮检测系统 =====")
    print(f"处理视频: {input_video}")
    print(f"输出视频: {output_video}")
    
    # 运行投篮检测系统
    detector = ShotDetector(input_video=input_video, output_video=output_video)
    
    # 等待一秒，确保文件已经写入磁盘
    time.sleep(1)
    
    # 查找最新生成的调试日志文件
    video_name = os.path.basename(input_video).split('.')[0]
    debug_logs = glob.glob(f"{video_name}_debug_*.json")
    
    if debug_logs:
        # 按修改时间排序，获取最新的日志文件
        latest_log = max(debug_logs, key=os.path.getmtime)
        print(f"找到调试日志文件: {latest_log}")
        return latest_log
    else:
        print("警告: 未找到调试日志文件")
        return None

def analyze_debug_log(log_file, output_dir):
    """
    分析调试日志
    
    Args:
        log_file: 调试日志文件路径
        output_dir: 分析结果输出目录
    """
    if not log_file or not os.path.exists(log_file):
        print("错误: 调试日志文件不存在")
        return
    
    print(f"\n===== 步骤 2: 分析调试日志 =====")
    print(f"分析日志文件: {log_file}")
    print(f"输出目录: {output_dir}")
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 调用分析脚本
    analyze_logs(['--log', log_file, '--output', output_dir])

def display_results(analysis_dir):
    """
    显示分析结果
    
    Args:
        analysis_dir: 分析结果目录
    """
    print(f"\n===== 步骤 3: 分析结果摘要 =====")
    
    # 检查分析目录是否存在
    if not os.path.exists(analysis_dir):
        print(f"错误: 分析结果目录 {analysis_dir} 不存在")
        return
    
    # 显示投篮摘要报告
    summary_file = os.path.join(analysis_dir, 'shot_summary.txt')
    if os.path.exists(summary_file):
        with open(summary_file, 'r') as f:
            summary = f.read()
        print("\n" + summary)
    else:
        print("警告: 未找到投篮摘要报告")
    
    # 列出生成的图表
    image_files = glob.glob(os.path.join(analysis_dir, '*.png'))
    if image_files:
        print("\n生成的可视化图表:")
        for img_file in image_files:
            print(f"- {os.path.basename(img_file)}")
    else:
        print("\n警告: 未找到可视化图表")
    
    print(f"\n所有分析结果已保存到: {os.path.abspath(analysis_dir)}")

def main():
    parser = argparse.ArgumentParser(description='篮球投篮检测系统调试功能演示')
    parser.add_argument('--input', type=str, required=True, help='输入视频文件路径')
    parser.add_argument('--output', type=str, help='输出视频文件路径')
    parser.add_argument('--analysis_dir', type=str, default='analysis_results', help='分析结果输出目录')
    args = parser.parse_args()
    
    # 如果未指定输出视频路径，则生成一个默认路径
    if not args.output:
        input_name = os.path.basename(args.input)
        input_base = os.path.splitext(input_name)[0]
        args.output = f"{input_base}_debug_output.mp4"
    
    # 运行投篮检测系统
    start_time = time.time()
    debug_log = run_shot_detection(args.input, args.output)
    
    if debug_log:
        # 分析调试日志
        analyze_debug_log(debug_log, args.analysis_dir)
        
        # 显示分析结果
        display_results(args.analysis_dir)
    
    # 显示总运行时间
    total_time = time.time() - start_time
    print(f"\n总运行时间: {total_time:.2f} 秒")
    print("\n演示完成!")

if __name__ == "__main__":
    main()