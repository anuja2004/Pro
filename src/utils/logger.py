"""
Logging setup for federated learning
"""
import logging
import sys
from datetime import datetime

def setup_logger(name, log_file=None, level=logging.INFO):
    """
    Set up logger with console and file handlers
    
    Args:
        name: Logger name
        log_file: Path to log file (optional)
        level: Logging level
        
    Returns:
        Configured logger
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers
    logger.handlers = []
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if log_file provided)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

class TrainingLogger:
    """Logger for training progress"""
    
    def __init__(self, log_file=None):
        self.logger = setup_logger('training', log_file)
        self.history = []
    
    def log_round(self, round_num, metrics):
        """Log metrics for a training round"""
        message = f"Round {round_num}: "
        for key, value in metrics.items():
            if isinstance(value, float):
                message += f"{key}={value:.4f} "
            else:
                message += f"{key}={value} "
        
        self.logger.info(message)
        self.history.append({
            'round': round_num,
            'timestamp': datetime.now().isoformat(),
            **metrics
        })
    
    def log_final(self, final_metrics):
        """Log final results"""
        self.logger.info("="*50)
        self.logger.info("FINAL RESULTS")
        self.logger.info("="*50)
        
        for client_name, metrics in final_metrics.items():
            self.logger.info(f"\n{client_name}:")
            for metric_name, value in metrics.items():
                if isinstance(value, float):
                    self.logger.info(f"  {metric_name}: {value:.4f}")
                else:
                    self.logger.info(f"  {metric_name}: {value}")
    
    def get_history(self):
        """Get training history"""
        return self.history
