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
        "speed": "[5 Star] Fastest",
        "accuracy": "[3 Star] Good",
        "memory": "[5 Star] Lowest"
    },
    "yolov11_small": {
        "ball_model": "Yolo-Weights/yolo11s.pt", 
        "description": "YOLOv11 Small - Good balance",
        "speed": "[4 Star] Fast",
        "accuracy": "[4 Star] Better",
        "memory": "[4 Star] Low"
    },
    "yolov11_medium": {
        "ball_model": "Yolo-Weights/yolo11m.pt",
        "description": "YOLOv11 Medium - Better than YOLOv8m",
        "speed": "[3 Star] Medium",
        "accuracy": "[4 Star] Better",
        "memory": "[3 Star] Medium"
    },
    "yolov11_large": {
        "ball_model": "Yolo-Weights/yolo11l.pt",
        "description": "YOLOv11 Large - High accuracy",
        "speed": "[3 Star] Medium",
        "accuracy": "[5 Star] Best",
        "memory": "[2 Star] High"
    },
    "yolov11_xlarge": {
        "ball_model": "Yolo-Weights/yolo11x.pt",
        "description": "YOLOv11 XLarge - Highest accuracy",
        "speed": "[2 Star] Slower",
        "accuracy": "[5 Star] Best",
        "memory": "[1 Star] Highest"
    },
    
    # YOLOv10 Series (Speed Optimized)
    "yolov10_medium": {
        "ball_model": "Yolo-Weights/yolov10m.pt",
        "description": "YOLOv10 Medium - Speed optimized",
        "speed": "[5 Star] Fastest",
        "accuracy": "[4 Star] Better",
        "memory": "[3 Star] Medium"
    },
    "yolov10_large": {
        "ball_model": "Yolo-Weights/yolov10l.pt",
        "description": "YOLOv10 Large - Fast + accurate",
        "speed": "[4 Star] Fast",
        "accuracy": "[5 Star] Best",
        "memory": "[2 Star] High"
    },
    
    # RT-DETR Series (Transformer-based)
    "rtdetr_large": {
        "ball_model": "Yolo-Weights/rtdetr-l.pt",
        "description": "RT-DETR Large - Transformer architecture",
        "speed": "[3 Star] Medium",
        "accuracy": "[5 Star] Best",
        "memory": "[2 Star] High"
    },
    "rtdetr_xlarge": {
        "ball_model": "Yolo-Weights/rtdetr-x.pt", 
        "description": "RT-DETR XLarge - Highest transformer accuracy",
        "speed": "[2 Star] Slower",
        "accuracy": "[5 Star] Best",
        "memory": "[1 Star] Highest"
    },
    
    # YOLOv8 Enhanced (Better than current)
    "yolov8_large": {
        "ball_model": "Yolo-Weights/yolov8l.pt",
        "description": "YOLOv8 Large - Better than current yolov8m",
        "speed": "[3 Star] Medium",
        "accuracy": "[4 Star] Better",
        "memory": "[2 Star] High"
    },
    "yolov8_xlarge": {
        "ball_model": "Yolo-Weights/yolov8x.pt",
        "description": "YOLOv8 XLarge - Highest YOLOv8 accuracy",
        "speed": "[2 Star] Slower",
        "accuracy": "[5 Star] Best",
        "memory": "[1 Star] Highest"
    },
    
    # Current baseline
    "current": {
        "ball_model": "Yolo-Weights/yolov8m.pt",
        "description": "Current YOLOv8 Medium (baseline)",
        "speed": "[3 Star] Medium",
        "accuracy": "[3 Star] Good",
        "memory": "[3 Star] Medium"
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
        from debug_logger import DebugLogger
        debug_logger = DebugLogger("model_configs")
        debug_logger.warning(f"Unknown config: {config_name}")
        debug_logger.info(f"Available configs: {list(MODEL_CONFIGS.keys())}")
        debug_logger.info(f"Recommended configs: {list(RECOMMENDED_CONFIGS.keys())}")
        return None

def list_all_configs():
    """List all available model configurations"""
    from debug_logger import DebugLogger
    debug_logger = DebugLogger("model_configs")
    debug_logger.info("\n[Basketball] Available Basketball Detection Models:\n")
    
    for name, config in MODEL_CONFIGS.items():
        debug_logger.info(f"[Config] {name}:")
        debug_logger.info(f"   Model: {config['ball_model']}")
        debug_logger.info(f"   Description: {config['description']}")
        debug_logger.info(f"   Speed: {config['speed']}")
        debug_logger.info(f"   Accuracy: {config['accuracy']}")
        debug_logger.info(f"   Memory: {config['memory']}")
        debug_logger.info("")
    
    debug_logger.info("[Recommendations] Recommended Configurations:")
    for scenario, config_name in RECOMMENDED_CONFIGS.items():
        config = MODEL_CONFIGS[config_name]
        debug_logger.info(f"   {scenario}: {config_name} ({config['ball_model']})")

if __name__ == "__main__":
    list_all_configs()
