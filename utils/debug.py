"""
Debug utilities for error handling and recovery
"""
import sys
import logging
import traceback
import subprocess
import psutil
from datetime import datetime


class DebugLogger:
    """Lightweight debug logging and error recovery"""
    
    def __init__(self, log_file="debug.log"):
        self.log_file = log_file
        self.setup_logging()
        
    def setup_logging(self):
        """Setup basic logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def log_error(self, error, context=""):
        """Log error with context"""
        self.logger.error(f"{context}: {error}")
        self.logger.error(traceback.format_exc())
        
    def log_system_info(self):
        """Log system information"""
        self.logger.info(f"Python: {sys.version}")
        self.logger.info(f"Platform: {sys.platform}")
        if hasattr(psutil, 'virtual_memory'):
            mem = psutil.virtual_memory()
            self.logger.info(f"Memory: {mem.total / 1e9:.1f}GB total, {mem.available / 1e9:.1f}GB available")
            
    def check_memory(self, threshold_gb=2.0):
        """Check if available memory is below threshold"""
        mem = psutil.virtual_memory()
        available_gb = mem.available / 1e9
        if available_gb < threshold_gb:
            self.logger.warning(f"Low memory: {available_gb:.1f}GB available")
            return False
        return True
        
    def install_missing_package(self, package):
        """Try to install missing package"""
        try:
            self.logger.info(f"Attempting to install {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            return True
        except:
            self.logger.error(f"Failed to install {package}")
            return False