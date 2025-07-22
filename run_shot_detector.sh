#!/bin/bash

# 脚本说明：这个脚本用于简化 shot_detector.py 的执行
# 使用方法：./run_shot_detector.sh [输入视频路径] [输出视频路径(可选)]

# 获取脚本所在目录的绝对路径
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# 默认输入视频路径
INPUT_VIDEO="video_test_5.mp4"
# 默认输出视频路径
OUTPUT_VIDEO=""

# 处理命令行参数
if [ $# -ge 1 ]; then
    INPUT_VIDEO="$1"
fi

if [ $# -ge 2 ]; then
    OUTPUT_VIDEO="--output $2"
fi

# 切换到脚本所在目录
cd "$SCRIPT_DIR"

# 执行命令
echo "正在处理视频: $INPUT_VIDEO"
if [ -n "$OUTPUT_VIDEO" ]; then
    echo "输出视频将保存为: $2"
fi
echo "日志文件将以视频名称和时间戳命名，格式为: <视频名称>_shot_log_<时间戳>.json"

xvfb-run --auto-servernum --server-args="-screen 0 1280x1024x24" python shot_detector.py --input "$INPUT_VIDEO" $OUTPUT_VIDEO

echo "处理完成"
echo "日志文件已保存在当前目录中"