"""
Basketball Detection Model Configurations
Provides different model options for optimal basketball detection
"""

# Model configurations for different use cases
MODEL_CONFIGS = {
    # YOLOv11 Series (Latest, Best Performance)
    "yolov11_nano": {
        "ball_model": "Yolo-Weights/yolo11n.pt",
        "description": "YOLOv11 Nano - Fastest, lowest memory",
        "speed": "⭐⭐⭐⭐⭐",
        "accuracy": "⭐⭐⭐",
        "memory": "⭐⭐⭐⭐⭐"
    },
    "yolov11_small": {
        "ball_model": "Yolo-Weights/yolo11s.pt", 
        "description": "YOLOv11 Small - Good balance",
        "speed": "⭐⭐⭐⭐",
        "accuracy": "⭐⭐⭐⭐",
        "memory": "⭐⭐⭐⭐"
    },
    "yolov11_medium": {
        "ball_model": "Yolo-Weights/yolo11m.pt",
        "description": "YOLOv11 Medium - Better than YOLOv8m",
        "speed": "⭐⭐⭐",
        "accuracy": "⭐⭐⭐⭐",
        "memory": "⭐⭐⭐"
    },
    "yolov11_large": {
        "ball_model": "Yolo-Weights/yolo11l.pt",
        "description": "YOLOv11 Large - High accuracy",
        "speed": "⭐⭐⭐",
        "accuracy": "⭐⭐⭐⭐⭐",
        "memory": "⭐⭐"
    },
    "yolov11_xlarge": {
        "ball_model": "Yolo-Weights/yolo11x.pt",
        "description": "YOLOv11 XLarge - Highest accuracy",
        "speed": "⭐⭐",
        "accuracy": "⭐⭐⭐⭐⭐",
        "memory": "⭐"
    },
    
    # YOLOv10 Series (Speed Optimized)
    "yolov10_medium": {
        "ball_model": "Yolo-Weights/yolov10m.pt",
        "description": "YOLOv10 Medium - Speed optimized",
        "speed": "⭐⭐⭐⭐⭐",
        "accuracy": "⭐⭐⭐⭐",
        "memory": "⭐⭐⭐"
    },
    "yolov10_large": {
        "ball_model": "Yolo-Weights/yolov10l.pt",
        "description": "YOLOv10 Large - Fast + accurate",
        "speed": "⭐⭐⭐⭐",
        "accuracy": "⭐⭐⭐⭐⭐",
        "memory": "⭐⭐"
    },
    
    # RT-DETR Series (Transformer-based)
    "rtdetr_large": {
        "ball_model": "Yolo-Weights/rtdetr-l.pt",
        "description": "RT-DETR Large - Transformer architecture",
        "speed": "⭐⭐⭐",
        "accuracy": "⭐⭐⭐⭐⭐",
        "memory": "⭐⭐"
    },
    "rtdetr_xlarge": {
        "ball_model": "Yolo-Weights/rtdetr-x.pt", 
        "description": "RT-DETR XLarge - Highest transformer accuracy",
        "speed": "⭐⭐",
        "accuracy": "⭐⭐⭐⭐⭐",
        "memory": "⭐"
    },
    
    # YOLOv8 Enhanced (Better than current)
    "yolov8_large": {
        "ball_model": "Yolo-Weights/yolov8l.pt",
        "description": "YOLOv8 Large - Better than current yolov8m",
        "speed": "⭐⭐⭐",
        "accuracy": "⭐⭐⭐⭐",
        "memory": "⭐⭐"
    },
    "yolov8_xlarge": {
        "ball_model": "Yolo-Weights/yolov8x.pt",
        "description": "YOLOv8 XLarge - Highest YOLOv8 accuracy",
        "speed": "⭐⭐",
        "accuracy": "⭐⭐⭐⭐⭐",
        "memory": "⭐"
    },
    
    # Current baseline
    "current": {
        "ball_model": "Yolo-Weights/yolov8m.pt",
        "description": "Current YOLOv8 Medium (baseline)",
        "speed": "⭐⭐⭐",
        "accuracy": "⭐⭐⭐",
        "memory": "⭐⭐⭐"
    }
}

# Recommended configurations for different scenarios
RECOMMENDED_CONFIGS = {
    "real_time": "yolov11_small",      # For real-time processing
    "high_accuracy": "yolov11_xlarge", # For highest accuracy
    "balanced": "yolov11_medium",      # Best balance
    "speed_first": "yolov10_medium",   # Speed is priority
    "transformer": "rtdetr_large"      # Try transformer architecture
}

def get_model_config(config_name):
    """Get model configuration by name"""
    if config_name in MODEL_CONFIGS:
        return MODEL_CONFIGS[config_name]
    elif config_name in RECOMMENDED_CONFIGS:
        return MODEL_CONFIGS[RECOMMENDED_CONFIGS[config_name]]
    else:
        print(f"Unknown config: {config_name}")
        print("Available configs:", list(MODEL_CONFIGS.keys()))
        print("Recommended configs:", list(RECOMMENDED_CONFIGS.keys()))
        return None

def list_all_configs():
    """List all available model configurations"""
    print("\n🏀 Available Basketball Detection Models:\n")
    
    for name, config in MODEL_CONFIGS.items():
        print(f"📋 {name}:")
        print(f"   Model: {config['ball_model']}")
        print(f"   Description: {config['description']}")
        print(f"   Speed: {config['speed']}")
        print(f"   Accuracy: {config['accuracy']}")
        print(f"   Memory: {config['memory']}")
        print()
    
    print("🎯 Recommended Configurations:")
    for scenario, config_name in RECOMMENDED_CONFIGS.items():
        config = MODEL_CONFIGS[config_name]
        print(f"   {scenario}: {config_name} ({config['ball_model']})")

if __name__ == "__main__":
    list_all_configs()
