import logging
import os
import sys
from datetime import datetime
from typing import Optional
import json


class ColoredFormatter(logging.Formatter):
    """带颜色的日志格式化器"""
    
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record):
        log_color = self.COLORS.get(record.levelname, '')
        record.levelname = f"{log_color}{record.levelname}{self.RESET}"
        return super().format(record)


def setup_logger(name: str = "UDANet",
                log_file: Optional[str] = None,
                level: str = "INFO",
                console: bool = True,
                file: bool = True) -> logging.Logger:
    """
    设置日志器
    
    Args:
        name: 日志器名称
        log_file: 日志文件路径
        level: 日志级别
        console: 是否输出到控制台
        file: 是否输出到文件
    
    Returns:
        配置好的日志器
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # 清除之前的处理器
    logger.handlers.clear()
    
    # 创建格式化器
    formatter = ColoredFormatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 控制台处理器
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, level.upper()))
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # 文件处理器
    if file and log_file:
        # 确保日志目录存在
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(getattr(logging, level.upper()))
        
        # 文件日志不使用颜色
        plain_formatter = logging.Formatter(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(plain_formatter)
        logger.addHandler(file_handler)
    
    return logger


class ExperimentLogger:
    """实验日志记录器"""
    
    def __init__(self, experiment_name: str, log_dir: str = "logs"):
        self.experiment_name = experiment_name
        self.log_dir = log_dir
        self.start_time = datetime.now()
        
        # 创建实验日志目录
        self.experiment_dir = os.path.join(log_dir, experiment_name)
        os.makedirs(self.experiment_dir, exist_ok=True)
        
        # 设置主日志器
        self.logger = setup_logger(
            name=experiment_name,
            log_file=os.path.join(self.experiment_dir, "experiment.log"),
            level="INFO"
        )
        
        # 实验数据
        self.metrics_history = {}
        self.config = {}
        self.notes = []
    
    def log_config(self, config: dict):
        """记录实验配置"""
        self.config = config
        config_file = os.path.join(self.experiment_dir, "config.json")
        
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Configuration saved to {config_file}")
        self.logger.info("Experiment configuration:")
        for key, value in config.items():
            self.logger.info(f"  {key}: {value}")
    
    def log_metrics(self, metrics: dict, step: int = None, prefix: str = ""):
        """记录指标"""
        metric_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        
        if step is not None:
            self.logger.info(f"Step {step} - {prefix}{metric_str}")
        else:
            self.logger.info(f"{prefix}{metric_str}")
        
        # 保存到历史记录
        for key, value in metrics.items():
            if key not in self.metrics_history:
                self.metrics_history[key] = []
            self.metrics_history[key].append(value)
    
    def log_note(self, note: str):
        """记录实验笔记"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        note_entry = f"[{timestamp}] {note}"
        self.notes.append(note_entry)
        self.logger.info(f"Note: {note}")
    
    def log_model_info(self, model):
        """记录模型信息"""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        self.logger.info("Model Information:")
        self.logger.info(f"  Total parameters: {total_params:,}")
        self.logger.info(f"  Trainable parameters: {trainable_params:,}")
        self.logger.info(f"  Model architecture: {model.__class__.__name__}")
    
    def save_metrics_history(self):
        """保存指标历史"""
        metrics_file = os.path.join(self.experiment_dir, "metrics_history.json")
        
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(self.metrics_history, f, indent=2)
        
        self.logger.info(f"Metrics history saved to {metrics_file}")
    
    def save_notes(self):
        """保存实验笔记"""
        notes_file = os.path.join(self.experiment_dir, "notes.txt")
        
        with open(notes_file, 'w', encoding='utf-8') as f:
            f.write("Experiment Notes\n")
            f.write("=" * 50 + "\n")
            f.write(f"Experiment: {self.experiment_name}\n")
            f.write(f"Start Time: {self.start_time}\n")
            f.write(f"End Time: {datetime.now()}\n")
            f.write("=" * 50 + "\n\n")
            
            for note in self.notes:
                f.write(f"{note}\n")
        
        self.logger.info(f"Notes saved to {notes_file}")
    
    def finish_experiment(self):
        """结束实验"""
        self.save_metrics_history()
        self.save_notes()
        
        duration = datetime.now() - self.start_time
        self.logger.info(f"Experiment completed in {duration}")
        self.logger.info(f"Results saved to {self.experiment_dir}")


class TrainingLogger:
    """训练过程日志记录器"""
    
    def __init__(self, log_dir: str = "logs", experiment_name: str = None):
        if experiment_name is None:
            experiment_name = f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.experiment_logger = ExperimentLogger(experiment_name, log_dir)
        self.epoch = 0
        self.best_metrics = {}
    
    def log_epoch_start(self, epoch: int, total_epochs: int):
        """记录epoch开始"""
        self.epoch = epoch
        self.experiment_logger.logger.info(f"Epoch {epoch + 1}/{total_epochs} started")
    
    def log_epoch_end(self, epoch: int, metrics: dict, duration: float):
        """记录epoch结束"""
        self.experiment_logger.log_metrics(metrics, step=epoch)
        self.experiment_logger.logger.info(
            f"Epoch {epoch + 1} completed in {duration:.2f}s"
        )
        
        # 更新最佳指标
        for key, value in metrics.items():
            if key not in self.best_metrics or value < self.best_metrics[key]:
                self.best_metrics[key] = value
                self.experiment_logger.logger.info(
                    f"New best {key}: {value:.4f}"
                )
    
    def log_validation(self, metrics: dict):
        """记录验证结果"""
        self.experiment_logger.log_metrics(metrics, step=self.epoch, prefix="Val ")
    
    def log_training(self, metrics: dict):
        """记录训练结果"""
        self.experiment_logger.log_metrics(metrics, step=self.epoch, prefix="Train ")
    
    def finish_training(self):
        """结束训练"""
        self.experiment_logger.logger.info("Training completed!")
        self.experiment_logger.logger.info("Best metrics:")
        for key, value in self.best_metrics.items():
            self.experiment_logger.logger.info(f"  {key}: {value:.4f}")
        
        self.experiment_logger.finish_experiment()


def get_logger(name: str = "PestDetection") -> logging.Logger:
    """获取默认日志器"""
    return logging.getLogger(name)


def log_system_info(logger: logging.Logger):
    """记录系统信息"""
    import platform
    import torch
    
    logger.info("System Information:")
    logger.info(f"  Platform: {platform.system()} {platform.release()}")
    logger.info(f"  Python: {platform.python_version()}")
    logger.info(f"  PyTorch: {torch.__version__}")
    
    if torch.cuda.is_available():
        logger.info(f"  CUDA: {torch.version.cuda}")
        logger.info(f"  GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        logger.info("  CUDA: Not available")


def log_model_summary(logger: logging.Logger, model, input_size: tuple):
    """记录模型摘要信息"""
    import torch
    
    logger.info("Model Summary:")
    logger.info(f"  Model Type: {model.__class__.__name__}")
    logger.info(f"  Input Size: {input_size}")
    
    # 计算参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"  Total Parameters: {total_params:,}")
    logger.info(f"  Trainable Parameters: {trainable_params:,}")
    logger.info(f"  Parameter Ratio: {trainable_params/total_params*100:.1f}%")
    
    # 估算模型大小
    param_size = total_params * 4  # 假设float32
    logger.info(f"  Estimated Model Size: {param_size / 1024**2:.1f} MB")


def log_dataset_info(logger: logging.Logger, train_size: int, val_size: int, test_size: int):
    """记录数据集信息"""
    total_size = train_size + val_size + test_size
    
    logger.info("Dataset Information:")
    logger.info(f"  Total Samples: {total_size:,}")
    logger.info(f"  Training Samples: {train_size:,} ({train_size/total_size*100:.1f}%)")
    logger.info(f"  Validation Samples: {val_size:,} ({val_size/total_size*100:.1f}%)")
    logger.info(f"  Test Samples: {test_size:,} ({test_size/total_size*100:.1f}%)")