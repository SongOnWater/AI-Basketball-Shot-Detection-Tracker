#!/usr/bin/env python3
"""
Frame Data Extractor Tool for Basketball Shot Detection
从篮球投篮检测日志中提取特定帧的数据

Usage:
    python extract_frame_data.py <log_file> <frame_number>
"""

import json
import argparse
import os

def extract_frame_data(log_file, frame_number):
    """
    从日志文件中提取指定帧的数据
    
    Args:
        log_file: 日志文件路径
        frame_number: 目标帧号
    
    Returns:
        指定帧的完整数据（字典格式），若未找到则返回None
    """
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            frame_log = json.load(f)
            
        for frame_data in frame_log:
            if frame_data.get('frame') == frame_number:
                return frame_data
                
        print(f"⚠️ 未找到帧 {frame_number} 的数据")
        return None
        
    except FileNotFoundError:
        print(f"❌ 错误：文件 {log_file} 不存在")
        return None
    except json.JSONDecodeError as e:
        print(f"❌ 错误：JSON格式无效 - {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='从篮球投篮检测日志中提取特定帧的数据')
    parser.add_argument('log_file', help='日志文件路径（如720_0_frame_log_20250723_145845.json）')
    parser.add_argument('frame_number', type=int, help='目标帧号')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.log_file):
        print(f"❌ 错误：文件 {args.log_file} 不存在")
        return
    
    frame_data = extract_frame_data(args.log_file, args.frame_number)
    
    if frame_data:
        print(f"\n🔄 帧 {args.frame_number} 的数据：")
        print(json.dumps(frame_data, indent=4, ensure_ascii=False))

if __name__ == "__main__":
    main()