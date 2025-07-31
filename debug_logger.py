"""
Basketball Shot Detection Debug Logger
Provides structured logging with frame count context
"""

import os
import logging
import datetime
import json
from pathlib import Path

class DebugLogger:
    """
    Enhanced debug logger for basketball shot detection
    - Supports console and file logging
    - Includes frame count in log messages
    - Supports different log levels
    - Can filter console output while maintaining full file logs
    """
    
    def __init__(self, name, log_dir="logs", console_level=logging.INFO, file_level=logging.DEBUG):
        """
        Initialize debug logger
        
        Args:
            name: Logger name
            log_dir: Directory for log files
            console_level: Logging level for console output
            file_level: Logging level for file output
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)  # Capture all levels
        self.logger.propagate = False
        
        # Clear any existing handlers
        if self.logger.handlers:
            self.logger.handlers.clear()
            
        # Create logs directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        # Generate timestamp for log file name
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.debug_log_file = os.path.join(log_dir, f"{name}_{timestamp}.log")
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(console_level)
        console_format = logging.Formatter('%(levelname)s: %(message)s')
        console_handler.setFormatter(console_format)
        self.logger.addHandler(console_handler)
        
        # Create file handler
        file_handler = logging.FileHandler(self.debug_log_file)
        file_handler.setLevel(file_level)
        file_format = logging.Formatter('%(asctime)s [%(levelname)s] [Frame %(frame_count)s] %(message)s')
        file_handler.setFormatter(file_format)
        self.logger.addHandler(file_handler)
        
        # Store handlers for selective logging
        self.console_handler = console_handler
        self.file_handler = file_handler
        
        # Log initialization
        self.info(f"Debug logger initialized. Log file: {self.debug_log_file}", frame_count=0)
    
    def _log(self, level, message, frame_count=None, **kwargs):
        """
        Internal logging method with frame count
        
        Args:
            level: Logging level
            message: Log message
            frame_count: Current frame count
            **kwargs: Additional context
        """
        # Ensure frame_count is included in extra
        extra = {'frame_count': frame_count if frame_count is not None else 0}
        
        # Add any additional kwargs to extra
        if kwargs:
            extra.update(kwargs)
            
        # Log with extra context
        self.logger.log(level, message, extra=extra)
    
    def debug(self, message, frame_count=None, **kwargs):
        """Log debug message"""
        self._log(logging.DEBUG, message, frame_count, **kwargs)
    
    def info(self, message, frame_count=None, **kwargs):
        """Log info message"""
        self._log(logging.INFO, message, frame_count, **kwargs)
    
    def warning(self, message, frame_count=None, **kwargs):
        """Log warning message"""
        self._log(logging.WARNING, message, frame_count, **kwargs)
    
    def error(self, message, frame_count=None, **kwargs):
        """Log error message"""
        self._log(logging.ERROR, message, frame_count, **kwargs)
    
    def critical(self, message, frame_count=None, **kwargs):
        """Log critical message"""
        self._log(logging.CRITICAL, message, frame_count, **kwargs)
    
    def debug_file_only(self, message, frame_count=None, **kwargs):
        """
        Log debug message to file only (no console output)
        Useful for high-volume debug messages
        """
        # Temporarily disable console handler
        self.console_handler.setLevel(logging.CRITICAL)
        self.debug(message, frame_count, **kwargs)
        # Restore console handler level
        self.console_handler.setLevel(logging.INFO)
    
    def close(self):
        """Close logger and handlers"""
        for handler in self.logger.handlers:
            handler.close()
            self.logger.removeHandler(handler)
            
    def set_console_level(self, level):
        """Set console output logging level"""
        self.console_handler.setLevel(level)
        
    def set_file_level(self, level):
        """Set file output logging level"""
        self.file_handler.setLevel(level)