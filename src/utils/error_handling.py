"""
Enhanced Error Handling ve Logging
Proje-wide error handling ve logging mekanizması
"""

import logging
import functools
import traceback
from pathlib import Path
from typing import Optional, Callable, Any
from datetime import datetime

# Logging configuration
LOG_DIR = Path('outputs/logs')
LOG_DIR.mkdir(parents=True, exist_ok=True)


class ProjectLogger:
    """Merkezi logging sistemi"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._setup_logging()
        return cls._instance
    
    def _setup_logging(self):
        """Logging'i konfigüre et"""
        self.logger = logging.getLogger('NeAR_Project')
        self.logger.setLevel(logging.DEBUG)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # File handler
        log_file = LOG_DIR / f"project_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)
    
    def get_logger(self, module_name: str):
        """Module-specific logger"""
        return self.logger.getChild(module_name)


def log_exceptions(module_name: str):
    """Decorator: Function exceptions'ı log et"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            logger = ProjectLogger().get_logger(module_name)
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(
                    f"Error in {func.__name__}: {str(e)}\n{traceback.format_exc()}"
                )
                raise  # Re-raise after logging
        return wrapper
    return decorator


class CustomException(Exception):
    """Base exception class"""
    def __init__(self, message: str, error_code: str = 'UNKNOWN'):
        self.message = message
        self.error_code = error_code
        super().__init__(self.message)
        
        logger = ProjectLogger().get_logger('exceptions')
        logger.error(f"[{error_code}] {message}")


class DataLoadingError(CustomException):
    """Veri yükleme hatası"""
    def __init__(self, message: str):
        super().__init__(message, error_code='DATA_LOAD_ERROR')


class ModelError(CustomException):
    """Model oluşturma/yükleme hatası"""
    def __init__(self, message: str):
        super().__init__(message, error_code='MODEL_ERROR')


class TrainingError(CustomException):
    """Training hatası"""
    def __init__(self, message: str):
        super().__init__(message, error_code='TRAINING_ERROR')


class ConfigError(CustomException):
    """Konfigürasyon hatası"""
    def __init__(self, message: str):
        super().__init__(message, error_code='CONFIG_ERROR')


# Usage examples
if __name__ == '__main__':
    logger = ProjectLogger().get_logger('test')
    
    @log_exceptions('test_module')
    def test_function():
        logger.info("Test starting...")
        raise ValueError("Test error")
    
    try:
        test_function()
    except Exception as e:
        logger.error(f"Caught: {e}")
    
    print("✅ Logging sistemi kuruldu!")
