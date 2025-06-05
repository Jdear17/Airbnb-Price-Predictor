import logging
import os
from src.config import config

def setup_logger(name: str) -> logging.Logger:
    """
    Set up and configure a logger with the specified name.
    
    Args:
        name (str): Name of the logger
        
    Returns:
        logging.Logger: Configured logger instance
    """
    # Create logs directory if it doesn't exist
    os.makedirs(os.path.dirname(config.LOG_FILE), exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(config.LOG_LEVEL)
    
    # Create handlers
    file_handler = logging.FileHandler(config.LOG_FILE)
    console_handler = logging.StreamHandler()
    
    # Create formatters and add it to handlers
    formatter = logging.Formatter(config.LOG_FORMAT)
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def log_model_metrics(logger: logging.Logger, metrics: dict):
    """
    Log model metrics in a structured way.
    
    Args:
        logger (logging.Logger): Logger instance
        metrics (dict): Dictionary containing model metrics
    """
    logger.info("Model Performance Metrics:")
    for metric_name, value in metrics.items():
        logger.info(f"{metric_name}: {value:.4f}") 