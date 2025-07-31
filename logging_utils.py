"""
日志工具模块，包含 DebugLogger 和 ShotLogger 类
"""
import os
import json
from datetime import datetime


class DebugLogger:
    """
    调试日志记录器，用于记录控制台输出(纯文本格式)
    """
    def __init__(self, debug_log_file=None, input_file=None, model_path=None, log_type="debug", console_enabled=True):
        if input_file and model_path:
            input_name = os.path.splitext(os.path.basename(input_file))[0]
            model_name = os.path.splitext(os.path.basename(model_path))[0]
            timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            self.debug_log_file = os.path.join(os.path.dirname(debug_log_file or ''), f'{input_name}_{model_name}_{log_type}_{timestamp}.txt')
        else:
            self.debug_log_file = debug_log_file or os.path.join('logs', 'debug_output.txt')
        
        self._log_file = None
        self._console_enabled = console_enabled  # 控制台输出开关

        if self.debug_log_file:
            os.makedirs(os.path.dirname(self.debug_log_file), exist_ok=True)
            self._log_file = open(self.debug_log_file, 'w', encoding='utf-8')

    def debug(self, message, frame_count=None):
        """记录调试信息"""
        self._log('DEBUG', message, console_output=False, frame_count=frame_count)

    def debug_file_only(self, message, frame_count=None):
        """仅写入文件的调试信息"""
        self._log('DEBUG', message, console_output=False, frame_count=frame_count)

    def console(self, message, frame_count=None):
        """记录控制台和文件都输出的重要信息"""
        self._log('CONSOLE', message, console_output=True, frame_count=frame_count)

    def info(self, message, frame_count=None):
        """记录一般信息"""
        self._log('INFO', message, console_output=True, frame_count=frame_count)

    def warning(self, message, frame_count=None):
        """记录警告信息"""
        self._log('WARNING', message, console_output=True, frame_count=frame_count)

    def _log(self, level, message, console_output=True, frame_count=None):
        """内部日志记录方法"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        
        # 添加帧序号（如果提供）
        if frame_count is not None:
            log_entry = f"[{timestamp}] [Frame:{frame_count}] [{level}] {message}\n"
        else:
            log_entry = f"[{timestamp}] [{level}] {message}\n"
        
        if console_output and self._console_enabled:
            pass
        
        if self._log_file:
            self._log_file.write(log_entry)

    def close(self):
        """关闭日志文件"""
        if self._log_file and not self._log_file.closed:
            self._log_file.close()


class ShotLogger:
    """
    投篮日志记录器，专门用于记录投篮相关数据
    """
    def __init__(self, log_dir='logs', input_file=None, model_path=None, log_type="frame"):
        self.input_file = input_file
        self.model_path = model_path
        self.log_type = log_type  # 添加日志类型属性
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        input_name = os.path.splitext(os.path.basename(input_file))[0] if input_file else 'unknown_input'
        model_name = os.path.splitext(os.path.basename(model_path))[0] if model_path and isinstance(model_path, str) else 'custom_model'
        self.log_file = os.path.join(log_dir, f'{input_name}_{model_name}_{log_type}_{timestamp}.json')
        self.frame_count = 0
        self._log_data = []
        # 添加统计数据
        self._stats = {
            "video_cuts": [],
            "up_down_pairs": []  # 修改为存储UP/DOWN对
        }
        self._current_up = None  # 临时存储当前UP事件

    def log_frame_data(self, frame_idx, ball_pos, hoop_pos, person_pos,
                      selected_ball_idx, selected_hoop_idx, selected_person_idx,
                      current_frame_balls, current_frame_hoops, current_frame_persons,
                      selected_ball, selected_hoop):
        """记录帧数据"""
        frame_data = {
            "frame_idx": frame_idx,
            "ball_positions": [{"x": p[0][0], "y": p[0][1], "frame": p[1], 
                              "width": p[2], "height": p[3], "confidence": p[4]} 
                             for p in ball_pos],
            "hoop_positions": [{"x": p[0][0], "y": p[0][1], "frame": p[1], 
                              "width": p[2], "height": p[3], "confidence": p[4]} 
                             for p in hoop_pos],
            "selected_ball_idx": selected_ball_idx,
            "selected_hoop_idx": selected_hoop_idx,
            "current_frame_balls": current_frame_balls,
            "current_frame_hoops": current_frame_hoops
        }
        
        if person_pos:
            frame_data["person_positions"] = [{"x": p[0][0], "y": p[0][1], "frame": p[1], 
                                            "width": p[2], "height": p[3], "confidence": p[4]} 
                                           for p in person_pos]
            frame_data["selected_person_idx"] = selected_person_idx
            frame_data["current_frame_persons"] = current_frame_persons
        
        if selected_ball:
            frame_data["selected_ball"] = {"x": selected_ball[0][0], "y": selected_ball[0][1], 
                                         "frame": selected_ball[1], "width": selected_ball[2], 
                                         "height": selected_ball[3], "confidence": selected_ball[4]}
        
        if selected_hoop:
            frame_data["selected_hoop"] = {"x": selected_hoop[0][0], "y": selected_hoop[0][1], 
                                         "frame": selected_hoop[1], "width": selected_hoop[2], 
                                         "height": selected_hoop[3], "confidence": selected_hoop[4]}
        
        self._log_data.append(frame_data)

    def log_shot(self, frame_idx, timestamp, ball_pos, hoop_pos, ball_confidence, is_successful, debug_info=None):
        """记录投篮结果"""
        shot_data = {
            "frame_idx": frame_idx,
            "timestamp": timestamp,
            "ball_pos": {"x": ball_pos[0], "y": ball_pos[1]},
            "hoop_pos": {"x": hoop_pos[0], "y": hoop_pos[1]},
            "ball_confidence": ball_confidence,
            "is_successful": is_successful,
            "debug_info": debug_info
        }
        self._log_data.append({"shot": shot_data})

    def log_video_cut(self, frame_idx):
        """记录视频剪辑事件"""
        self._stats["video_cuts"].append(frame_idx)

    def log_up_event(self, frame_idx):
        """记录UP事件"""
        self._current_up = frame_idx

    def log_down_event(self, frame_idx):
        """记录DOWN事件，并与之前的UP事件组成一对"""
        if self._current_up is not None:
            # 将UP/DOWN对添加到统计数据中
            self._stats["up_down_pairs"].append({
                "up_frame": self._current_up,
                "down_frame": frame_idx
            })
            self._current_up = None  # 重置当前UP事件
        else:
            # 如果没有对应的UP事件，则单独记录DOWN事件
            self._stats["up_down_pairs"].append({
                "up_frame": None,
                "down_frame": frame_idx
            })

    def finalize_incomplete_up(self):
        """处理最后可能未完成的UP事件"""
        if self._current_up is not None:
            # 如果存在未配对的UP事件，将其作为缺省DOWN的对记录
            self._stats["up_down_pairs"].append({
                "up_frame": self._current_up,
                "down_frame": None
            })
            self._current_up = None

    def update_progress(self, current_frame, total_frames):
        """更新处理进度"""
        self.frame_count = current_frame

    def save_log(self):
        """保存日志到文件"""
        # 处理可能未完成的UP事件
        self.finalize_incomplete_up()
        
        # 如果是投篮日志类型，添加统计数据
        if "shot" in self.log_file:
            # 计算统计数据
            shots = [entry for entry in self._log_data if "shot" in entry]
            total_shots = len(shots)
            successful_shots = sum(1 for shot in shots if shot["shot"]["is_successful"])
            failed_shots = total_shots - successful_shots
            success_rate = successful_shots / total_shots if total_shots > 0 else 0
            
            # 创建包含统计数据和详细投篮信息的结构
            log_content = {
                "statistics": {
                    "total_shots": total_shots,
                    "successful_shots": successful_shots,
                    "failed_shots": failed_shots,
                    "success_rate": success_rate,
                    "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                },
                "shots": [entry["shot"] for entry in self._log_data if "shot" in entry]
            }
            
            with open(self.log_file, 'w', encoding='utf-8') as f:
                json.dump(log_content, f, indent=2, ensure_ascii=False)
        else:
            # 帧日志添加统计数据
            log_content = {
                "statistics": {
                    "video_cuts": {
                        "count": len(self._stats["video_cuts"]),
                        "frames": self._stats["video_cuts"]
                    },
                    "up_down_pairs": {
                        "count": len(self._stats["up_down_pairs"]),
                        "pairs": self._stats["up_down_pairs"]
                    },
                    "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                },
                "frames": self._log_data
            }
            
            with open(self.log_file, 'w', encoding='utf-8') as f:
                json.dump(log_content, f, indent=2, ensure_ascii=False)
        return self.log_file

    def print_improved_summary(self,debug_logger):
        """打印改进的摘要信息"""
        shots = [entry for entry in self._log_data if "shot" in entry]
        makes = sum(1 for shot in shots if shot["shot"]["is_successful"])
        attempts = len(shots)
        debug_logger.info(f"\n[Basketball] 投篮统计: {makes}/{attempts} ({makes/attempts*100:.1f}%)", frame_count=0)
        if attempts > 0:
            debug_logger.info(f"[Made] 命中: {makes}", frame_count=0)
            debug_logger.info(f"[Missed] 未中: {attempts - makes}", frame_count=0)
        else:
            debug_logger.warning("[Warning] 没有检测到投篮尝试", frame_count=0)