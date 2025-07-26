#!/usr/bin/env python3
"""
YOLO模型测试工具
Test different YOLO models on specific video frames
"""

import cv2
import json
import time
import argparse
from pathlib import Path
from ultralytics import YOLO
import numpy as np
from datetime import datetime

class YOLOModelTester:
    def __init__(self, video_path=None, output_dir="yolo_test_results"):
        self.video_path = video_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # 支持的YOLO模型列表 - 从Yolo-Weights目录动态加载
        self.weights_dir = Path('Yolo-Weights')
        if not self.weights_dir.exists():
            print(f"警告: 模型目录 {self.weights_dir} 不存在，将创建该目录")
            self.weights_dir.mkdir(exist_ok=True)
            self.available_models = []
        else:
            # 从目录中加载所有.pt文件
            self.available_models = [f.name for f in self.weights_dir.glob('*.pt')]
            if not self.available_models:
                print(f"警告: 在 {self.weights_dir} 目录中未找到任何模型文件")
                self.available_models = []
        
        # 篮球相关的类别ID
        self.ball_classes = ['sports ball']
        self.hoop_classes = ['sports ball']  # YOLO没有专门的篮筐类别，通常需要自定义训练

    def list_available_models(self):
        """列出所有可用模型"""
        print("\n📋 可用的YOLO模型:")
        print("="*60)
        
        if not self.available_models:
            print("⚠️ 未找到任何模型文件，请检查 Yolo-Weights 目录")
            print(f"模型目录路径: {self.weights_dir.absolute()}")
            print("="*60)
            return

        yolo11_models = [m for m in self.available_models if m.startswith('yolo11')]
        yolo8_models = [m for m in self.available_models if m.startswith('yolov8')]
        other_models = [m for m in self.available_models if not (m.startswith('yolo11') or m.startswith('yolov8'))]

        if yolo11_models:
            print("YOLOv11 系列:")
            for model in yolo11_models:
                size = self._get_model_size_description(model)
                print(f"  - {model:<12} {size}")

        if yolo8_models:
            print("\nYOLOv8 系列:")
            for model in yolo8_models:
                size = self._get_model_size_description(model)
                print(f"  - {model:<12} {size}")
                
        if other_models:
            print("\n其他模型:")
            for model in other_models:
                print(f"  - {model}")

        print("="*60)
        print("💡 使用示例: --models yolo11l yolo11s")
        print(f"📂 模型目录: {self.weights_dir.absolute()}")

    def _get_model_size_description(self, model):
        """获取模型大小描述"""
        if 'n.pt' in model:
            return "(Nano - 最快)"
        elif 's.pt' in model:
            return "(Small - 快速)"
        elif 'm.pt' in model:
            return "(Medium - 平衡)"
        elif 'l.pt' in model:
            return "(Large - 高精度)"
        elif 'x.pt' in model:
            return "(Extra Large - 最高精度)"
        return ""
        
    def load_video(self):
        """加载视频文件"""
        if not self.video_path:
            raise ValueError("未指定视频文件路径")
            
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise ValueError(f"无法打开视频文件: {self.video_path}")
        
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        print(f"视频信息:")
        print(f"  总帧数: {self.total_frames}")
        print(f"  帧率: {self.fps:.2f} FPS")
        
    def get_frame(self, frame_index):
        """获取指定帧"""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = self.cap.read()
        if not ret:
            raise ValueError(f"无法读取第 {frame_index} 帧")
        return frame
        
    def test_model_on_frame(self, model_name, frame, frame_index):
        """在单帧上测试模型"""
        try:
            # 加载模型（从 Yolo-Weights 文件夹）
            model_path = self.weights_dir / model_name
            if not model_path.exists():
                print(f"错误: 模型文件 {model_path} 不存在")
                return None
                
            model = YOLO(str(model_path))
            
            # 记录推理时间
            start_time = time.time()
            results = model(frame, verbose=False)
            inference_time = time.time() - start_time
            
            # 解析结果
            detections = []
            if results[0].boxes is not None:
                boxes = results[0].boxes
                for i in range(len(boxes)):
                    box = boxes[i]
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    cls = int(box.cls[0].cpu().numpy())
                    class_name = model.names[cls]
                    
                    # 计算中心点和尺寸
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    width = x2 - x1
                    height = y2 - y1
                    
                    detection = {
                        'class': class_name,
                        'confidence': float(conf),
                        'bbox': [float(x1), float(y1), float(x2), float(y2)],
                        'center': [float(center_x), float(center_y)],
                        'size': [float(width), float(height)]
                    }
                    detections.append(detection)
            
            return {
                'model': model_name,
                'frame_index': frame_index,
                'inference_time': inference_time,
                'detections': detections,
                'total_detections': len(detections),
                'ball_detections': [d for d in detections if d['class'] in self.ball_classes],
                'success': True,
                'error': None
            }
            
        except Exception as e:
            return {
                'model': model_name,
                'frame_index': frame_index,
                'inference_time': 0,
                'detections': [],
                'total_detections': 0,
                'ball_detections': [],
                'success': False,
                'error': str(e)
            }
    
    def test_models_on_frames(self, frame_indices, models=None):
        """在指定帧上测试多个模型"""
        if models is None:
            models = self.available_models

        # 验证模型名称并过滤可用模型
        valid_models = []
        for model in models:
            # 支持简化名称（如 'yolo11l' -> 'yolo11l.pt'）
            if not model.endswith('.pt'):
                model = model + '.pt'

            # 检查模型文件是否存在
            model_path = self.weights_dir / model
            if model_path.exists():
                valid_models.append(model)
            else:
                print(f"⚠️  警告: 权重文件不存在 '{model_path}'，将跳过")

        if not valid_models:
            print("❌ 没有有效的模型可测试")
            return None

        self.load_video()

        results = {
            'video_path': str(self.video_path),
            'test_time': datetime.now().isoformat(),
            'total_frames': self.total_frames,
            'fps': self.fps,
            'tested_frames': frame_indices,
            'tested_models': valid_models,
            'results': []
        }

        print(f"\n开始测试 {len(valid_models)} 个模型在 {len(frame_indices)} 帧上的表现...")
        print(f"测试模型: {', '.join(valid_models)}")

        for frame_idx in frame_indices:
            print(f"\n测试第 {frame_idx} 帧:")

            try:
                frame = self.get_frame(frame_idx)
                frame_results = {
                    'frame_index': frame_idx,
                    'timestamp': frame_idx / self.fps,
                    'model_results': []
                }

                for model_name in valid_models:
                    print(f"  测试模型: {model_name}")
                    result = self.test_model_on_frame(model_name, frame, frame_idx)
                    frame_results['model_results'].append(result)

                    if result['success']:
                        ball_count = len(result['ball_detections'])
                        total_detections = result['total_detections']
                        print(f"    ✅ 推理时间: {result['inference_time']:.3f}s, 总检测: {total_detections}, 篮球: {ball_count}")

                        # 显示篮球检测的详细信息
                        for i, ball in enumerate(result['ball_detections']):
                            conf = ball['confidence']
                            center = ball['center']
                            print(f"       篮球{i+1}: 置信度={conf:.2f}, 位置=({center[0]:.0f}, {center[1]:.0f})")
                    else:
                        print(f"    ❌ 错误: {result['error']}")

                results['results'].append(frame_results)

            except Exception as e:
                print(f"  ❌ 处理第 {frame_idx} 帧时出错: {str(e)}")

        self.cap.release()
        return results
    
    def test_frame_range(self, start_frame, end_frame, step=1, models=None):
        """测试帧范围"""
        frame_indices = list(range(start_frame, min(end_frame + 1, self.total_frames), step))
        return self.test_models_on_frames(frame_indices, models)
    
    def save_results(self, results, filename=None):
        """保存测试结果"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"yolo_test_results_{timestamp}.json"
        
        output_path = self.output_dir / filename
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n结果已保存到: {output_path}")
        return output_path
    
    def generate_report(self, results):
        """生成测试报告"""
        print("\n" + "="*60)
        print("YOLO模型测试报告")
        print("="*60)

        # 统计每个模型的表现
        model_stats = {}
        for frame_result in results['results']:
            for model_result in frame_result['model_results']:
                model_name = model_result['model']
                if model_name not in model_stats:
                    model_stats[model_name] = {
                        'total_tests': 0,
                        'successful_tests': 0,
                        'total_inference_time': 0,
                        'total_detections': 0,
                        'total_ball_detections': 0,
                        'errors': []
                    }

                stats = model_stats[model_name]
                stats['total_tests'] += 1

                if model_result['success']:
                    stats['successful_tests'] += 1
                    stats['total_inference_time'] += model_result['inference_time']
                    stats['total_detections'] += model_result['total_detections']
                    stats['total_ball_detections'] += len(model_result['ball_detections'])
                else:
                    stats['errors'].append(model_result['error'])

        # 打印报告
        for model_name, stats in model_stats.items():
            print(f"\n模型: {model_name}")
            print(f"  测试次数: {stats['total_tests']}")
            print(f"  成功率: {stats['successful_tests']}/{stats['total_tests']} ({stats['successful_tests']/stats['total_tests']*100:.1f}%)")

            if stats['successful_tests'] > 0:
                avg_time = stats['total_inference_time'] / stats['successful_tests']
                avg_detections = stats['total_detections'] / stats['successful_tests']
                avg_balls = stats['total_ball_detections'] / stats['successful_tests']

                print(f"  平均推理时间: {avg_time:.3f}s")
                print(f"  平均检测数量: {avg_detections:.1f}")
                print(f"  平均篮球检测: {avg_balls:.1f}")

            if stats['errors']:
                print(f"  错误: {len(stats['errors'])} 次")

        # 生成性能排名
        print(f"\n📊 性能排名:")
        successful_models = [(name, stats) for name, stats in model_stats.items()
                           if stats['successful_tests'] > 0]

        if successful_models:
            # 按推理速度排序
            speed_ranking = sorted(successful_models,
                                 key=lambda x: x[1]['total_inference_time'] / x[1]['successful_tests'])
            print(f"\n⚡ 速度排名 (推理时间):")
            for i, (name, stats) in enumerate(speed_ranking, 1):
                avg_time = stats['total_inference_time'] / stats['successful_tests']
                print(f"  {i}. {name}: {avg_time:.3f}s")

            # 按篮球检测数量排序
            ball_ranking = sorted(successful_models,
                                key=lambda x: x[1]['total_ball_detections'] / x[1]['successful_tests'],
                                reverse=True)
            print(f"\n🏀 篮球检测排名:")
            for i, (name, stats) in enumerate(ball_ranking, 1):
                avg_balls = stats['total_ball_detections'] / stats['successful_tests']
                print(f"  {i}. {name}: {avg_balls:.1f} 个/帧")

    def save_visualization(self, results, frame_indices=None, max_frames=None, classify_by='models'):
        """保存可视化结果
        
        Args:
            results: 测试结果
            frame_indices: 要处理的帧索引列表
            max_frames: 最大处理帧数
            classify_by: 分类方式，'models'按模型分类，'frames'按帧分类
        """
        if frame_indices is None:
            frame_indices = [result['frame_index'] for result in results['results']]

        # 如果指定了最大帧数限制，则应用限制
        if max_frames is not None:
            frame_indices = frame_indices[:max_frames]
            if len(frame_indices) < len([result['frame_index'] for result in results['results']]):
                print(f"⚠️  为了避免生成过多文件，只保存前 {max_frames} 帧的可视化结果")

        self.load_video()

        print(f"📸 正在生成 {len(frame_indices)} 帧的可视化结果...")
        print(f"📁 分类方式: {'按模型分类' if classify_by == 'models' else '按帧分类'}")

        # 创建分类目录
        if classify_by == 'models':
            # 获取所有模型名称
            model_names = set()
            for frame_result in results['results']:
                for model_result in frame_result['model_results']:
                    if model_result['success']:
                        model_names.add(model_result['model'])
            
            # 为每个模型创建子目录
            for model_name in model_names:
                model_dir = self.output_dir / model_name.replace('.pt', '')
                model_dir.mkdir(exist_ok=True)
                print(f"  创建目录: {model_dir}")
        
        for i, frame_idx in enumerate(frame_indices, 1):
            print(f"  处理第 {i}/{len(frame_indices)} 帧 (帧索引: {frame_idx})")
            try:
                frame = self.get_frame(frame_idx)

                # 如果按帧分类，为每一帧创建子目录
                if classify_by == 'frames':
                    frame_dir = self.output_dir / f"frame_{frame_idx}"
                    frame_dir.mkdir(exist_ok=True)

                # 为每个模型创建可视化
                frame_result = next((r for r in results['results'] if r['frame_index'] == frame_idx), None)
                if not frame_result:
                    continue

                for model_result in frame_result['model_results']:
                    if not model_result['success']:
                        continue

                    vis_frame = frame.copy()
                    model_name = model_result['model']

                    # 绘制检测框
                    for detection in model_result['detections']:
                        if detection['class'] in self.ball_classes:
                            # 篮球用红色框
                            color = (0, 0, 255)
                            thickness = 3
                        else:
                            # 其他物体用绿色框
                            color = (0, 255, 0)
                            thickness = 2

                        x1, y1, x2, y2 = map(int, detection['bbox'])
                        cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, thickness)

                        # 添加标签
                        label = f"{detection['class']}: {detection['confidence']:.2f}"
                        cv2.putText(vis_frame, label, (x1, y1-10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                    # 添加模型信息
                    info_text = f"Model: {model_name} | Frame: {frame_idx} | Time: {model_result['inference_time']:.3f}s"
                    cv2.putText(vis_frame, info_text, (10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                    # 根据分类方式确定保存路径
                    if classify_by == 'models':
                        # 按模型分类
                        model_dir = self.output_dir / model_name.replace('.pt', '')
                        output_path = model_dir / f"frame_{frame_idx}.jpg"
                    else:
                        # 按帧分类
                        frame_dir = self.output_dir / f"frame_{frame_idx}"
                        output_path = frame_dir / f"{model_name.replace('.pt', '')}.jpg"
                    
                    cv2.imwrite(str(output_path), vis_frame)

            except Exception as e:
                print(f"保存第 {frame_idx} 帧可视化时出错: {str(e)}")

        self.cap.release()
        print(f"可视化结果已保存到: {self.output_dir}")

    def organize_existing_images(self, classify_by='models'):
        """对已生成的图片进行分类整理
        
        Args:
            classify_by: 分类方式，'models'按模型分类，'frames'按帧分类
        """
        import re
        import shutil
        
        # 查找所有已生成的图片
        image_pattern = re.compile(r'frame_(\d+)_(.+)\.jpg')
        images = list(self.output_dir.glob('frame_*_*.jpg'))
        
        if not images:
            print("❌ 未找到任何图片文件，请先生成图片或检查目录")
            return
            
        print(f"📸 找到 {len(images)} 张图片，开始按{'模型' if classify_by == 'models' else '帧'}分类...")
        
        # 创建临时目录存储图片信息
        frame_model_map = {}
        
        # 解析文件名，提取帧号和模型名
        for img_path in images:
            match = image_pattern.match(img_path.name)
            if match:
                frame_idx = int(match.group(1))
                model_name = match.group(2)
                
                if frame_idx not in frame_model_map:
                    frame_model_map[frame_idx] = []
                    
                frame_model_map[frame_idx].append((model_name, img_path))
        
        # 按分类方式组织图片
        if classify_by == 'models':
            # 按模型分类
            model_dirs = {}
            for frame_idx, model_images in frame_model_map.items():
                for model_name, img_path in model_images:
                    # 创建模型目录
                    if model_name not in model_dirs:
                        model_dir = self.output_dir / model_name.replace('.pt', '')
                        model_dir.mkdir(exist_ok=True)
                        model_dirs[model_name] = model_dir
                        print(f"  创建目录: {model_dir}")
                    
                    # 复制图片到模型目录
                    dest_path = model_dirs[model_name] / f"frame_{frame_idx}.jpg"
                    shutil.copy2(img_path, dest_path)
        else:
            # 按帧分类
            for frame_idx, model_images in frame_model_map.items():
                # 创建帧目录
                frame_dir = self.output_dir / f"frame_{frame_idx}"
                frame_dir.mkdir(exist_ok=True)
                print(f"  创建目录: {frame_dir}")
                
                # 复制图片到帧目录
                for model_name, img_path in model_images:
                    dest_path = frame_dir / f"{model_name.replace('.pt', '')}.jpg"
                    shutil.copy2(img_path, dest_path)
        
        print(f"✅ 图片分类完成，已按{'模型' if classify_by == 'models' else '帧'}组织到子目录中")
    """解析帧索引字符串"""
def parse_frame_indices(frame_str):
    """解析帧索引字符串"""
    indices = []
    for part in frame_str.split(','):
        part = part.strip()
        if '-' in part:
            start, end = map(int, part.split('-'))
            indices.extend(range(start, end + 1))
        else:
            indices.append(int(part))
    return sorted(set(indices))

def main():
    parser = argparse.ArgumentParser(
        description='测试不同YOLO模型在视频特定帧上的表现',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 测试特定模型在篮球关键帧上的表现
  python test_yolo_models.py video.mp4 --basketball-frames --models yolo11l

  # 测试多个模型
  python test_yolo_models.py video.mp4 --frames "112,220" --models yolo11s yolo11l yolo11x

  # 只测试一个模型
  python test_yolo_models.py video.mp4 --frames "112" --models yolo11l --visualize

  # 测试帧范围，使用特定模型
  python test_yolo_models.py video.mp4 --range 100 200 10 --models yolo11s
  
  # 按模型分类保存可视化结果
  python test_yolo_models.py video.mp4 --frames "100,200" --models yolo11s yolo11l --visualize --classify-by models
  
  # 按帧分类保存可视化结果
  python test_yolo_models.py video.mp4 --frames "100,200" --models yolo11s yolo11l --visualize --classify-by frames
  
  # 对已生成的图片进行分类整理（需要提供视频路径）
  python test_yolo_models.py video.mp4 --organize-images --classify-by models
  
  # 对已生成的图片进行分类整理（不需要提供视频路径）
  python test_yolo_models.py --organize-images --classify-by models --output yolo_test_results
        """)

    parser.add_argument('video', nargs='?', help='输入视频文件路径，使用 --organize-images 时可选')
    parser.add_argument('--frames', help='要测试的帧索引，支持格式: "100,200,300" 或 "100-120,200-220"')
    parser.add_argument('--range', nargs=3, metavar=('START', 'END', 'STEP'), type=int,
                       help='测试帧范围: 起始帧 结束帧 步长')
    parser.add_argument('--models', nargs='+',
                       help='要测试的模型列表，如: yolo11l yolo11s yolov8l (可省略.pt后缀)')
    parser.add_argument('--output', help='输出目录', default='yolo_test_results')
    parser.add_argument('--visualize', action='store_true', help='保存可视化结果')
    parser.add_argument('--max-vis-frames', type=int, default=None,
                       help='最大可视化帧数限制（默认无限制）')
    parser.add_argument('--basketball-frames', action='store_true',
                       help='测试篮球相关的关键帧 (112, 120, 220, 310)')
    parser.add_argument('--list-models', action='store_true', help='列出所有可用模型')
    parser.add_argument('--classify-by', choices=['models', 'frames'], default='models',
                       help='图片分类方式: 按模型(models)或按帧(frames)分类，默认按模型分类')
    parser.add_argument('--organize-images', action='store_true',
                       help='对已生成的图片进行分类整理，不运行模型测试')

    args = parser.parse_args()

    # 创建测试器
    tester = YOLOModelTester(args.video, args.output)

    # 如果只是列出模型，则显示后退出
    if args.list_models:
        tester.list_available_models()
        return
        
    # 如果只是整理已有图片，则执行整理后退出
    if args.organize_images:
        print(f"\n📂 开始整理已有图片，按{args.classify_by}分类...")
        tester.organize_existing_images(classify_by=args.classify_by)
        return

    # 检查视频路径是否提供
    if not args.video:
        parser.error("必须提供视频文件路径才能进行模型测试")
        
    # 加载视频以获取总帧数等信息
    tester.load_video()

    # 显示将要测试的模型
    if args.models:
        print(f"🎯 指定测试模型: {', '.join(args.models)}")
    else:
        print(f"🎯 将测试所有可用模型 ({len(tester.available_models)} 个)")

    # 确定要测试的帧
    if args.basketball_frames:
        # 基于之前分析的篮球关键帧
        frame_indices = [112, 120, 220, 310]
        print(f"🏀 测试篮球关键帧: {frame_indices}")
        results = tester.test_models_on_frames(frame_indices, args.models)
    elif args.frames:
        frame_indices = parse_frame_indices(args.frames)
        print(f"📋 测试指定帧: {frame_indices}")
        results = tester.test_models_on_frames(frame_indices, args.models)
    elif args.range:
        start, end, step = args.range
        print(f"📊 测试帧范围: {start}-{end} (步长: {step})")
        results = tester.test_frame_range(start, end, step, args.models)
    else:
        # 默认测试一些关键帧
        default_frames = [100, 200, 300]
        print(f"📋 未指定帧，将测试默认帧: {default_frames}")
        results = tester.test_models_on_frames(default_frames, args.models)

    # 检查是否有有效结果
    if results is None:
        print("❌ 测试失败，没有有效结果")
        return

    # 保存结果并生成报告
    tester.save_results(results)
    tester.generate_report(results)

    # 保存可视化结果
    if args.visualize:
        print("\n📸 生成可视化结果...")
        tester.save_visualization(results, max_frames=args.max_vis_frames, classify_by=args.classify_by)

if __name__ == "__main__":
    main()
