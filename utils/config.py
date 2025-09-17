import yaml
import os
from easydict import EasyDict as edict


def load_config(config_path):
    """
    加载配置文件
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        config: 配置对象
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    
    config = edict(config_dict)
    
    # 设置默认值
    config = set_default_values(config)
    
    return config


def set_default_values(config):
    """
    设置配置默认值
    
    Args:
        config: 配置对象
        
    Returns:
        config: 更新后的配置对象
    """
    # 数据配置默认值
    if 'data' not in config:
        config.data = edict()
    
    config.data.setdefault('root_path', './data')
    config.data.setdefault('image_size', [512, 512])
    config.data.setdefault('num_classes', 7)
    config.data.setdefault('train_split', 0.8)
    config.data.setdefault('val_split', 0.1)
    config.data.setdefault('test_split', 0.1)
    
    # 模型配置默认值
    if 'model' not in config:
        config.model = edict()
    
    config.model.setdefault('name', 'UnifiedNet')
    config.model.setdefault('backbone', 'SwinTransformer')
    config.model.setdefault('pretrained', True)
    
    # 训练配置默认值
    if 'training' not in config:
        config.training = edict()
    
    config.training.setdefault('batch_size', 8)
    config.training.setdefault('num_epochs', 200)
    config.training.setdefault('num_workers', 4)
    config.training.setdefault('device', 'cuda')
    
    # 优化器配置默认值
    if 'optimizer' not in config:
        config.optimizer = edict()
    
    config.optimizer.setdefault('name', 'Adam')
    config.optimizer.setdefault('lr', 1e-4)
    config.optimizer.setdefault('weight_decay', 1e-5)
    
    # 日志配置默认值
    if 'logging' not in config:
        config.logging = edict()
    
    config.logging.setdefault('log_dir', './logs')
    config.logging.setdefault('log_freq', 10)
    config.logging.setdefault('tensorboard', True)
    
    return config


def merge_config(config, args):
    """
    合并配置文件和命令行参数
    
    Args:
        config: 配置对象
        args: 命令行参数
        
    Returns:
        config: 合并后的配置对象
    """
    # 合并训练参数
    if hasattr(args, 'batch_size') and args.batch_size is not None:
        config.training.batch_size = args.batch_size
    
    if hasattr(args, 'num_epochs') and args.num_epochs is not None:
        config.training.num_epochs = args.num_epochs
    
    if hasattr(args, 'lr') and args.lr is not None:
        config.optimizer.lr = args.lr
    
    if hasattr(args, 'device') and args.device is not None:
        config.training.device = args.device
    
    # 合并模型参数
    if hasattr(args, 'model_name') and args.model_name is not None:
        config.model.name = args.model_name
    
    if hasattr(args, 'backbone') and args.backbone is not None:
        config.model.backbone = args.backbone
    
    # 合并数据参数
    if hasattr(args, 'data_root') and args.data_root is not None:
        config.data.root_path = args.data_root
    
    if hasattr(args, 'image_size') and args.image_size is not None:
        config.data.image_size = args.image_size
    
    # 合并检查点参数
    if hasattr(args, 'resume') and args.resume is not None:
        config.checkpoint.resume = args.resume
    
    if hasattr(args, 'save_dir') and args.save_dir is not None:
        config.checkpoint.save_dir = args.save_dir
    
    return config


def save_config(config, save_path):
    """
    保存配置到文件
    
    Args:
        config: 配置对象
        save_path: 保存路径
    """
    # 将EasyDict转换为普通dict
    config_dict = dict(config)
    
    # 确保目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'w', encoding='utf-8') as f:
        yaml.dump(config_dict, f, default_flow_style=False, indent=2)


def print_config(config):
    """
    打印配置信息
    
    Args:
        config: 配置对象
    """
    print("=" * 60)
    print("Configuration:")
    print("=" * 60)
    
    def print_dict(d, indent=0):
        for key, value in d.items():
            if isinstance(value, dict):
                print("  " * indent + f"{key}:")
                print_dict(value, indent + 1)
            else:
                print("  " * indent + f"{key}: {value}")
    
    print_dict(config)
    print("=" * 60)


class Config:
    """Configuration class"""
    
    # Model configuration
    model_name = "UDANet"
    backbone = "resnet50"
    num_classes = 2  # low density, high density
    
    # Input configuration
    input_size = (512, 512)
    
    # Training configuration
    batch_size = 8
    num_epochs = 100
    learning_rate = 1e-4
    weight_decay = 1e-4
    
    # Density level threshold
    density_threshold = 10  # less than 10 points is low density
    
    # Data augmentation
    use_augmentation = True
    
    # Loss weights
    detection_loss_weight = 1.0
    density_loss_weight = 0.5
    
    # Save configuration
    save_dir = "checkpoints"
    log_dir = "logs"
    
    # Device configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def __init__(self):
        """Initialize configuration"""
        # Create save directories
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Set random seed
        self.set_seed(42)
    
    def set_seed(self, seed):
        """Set random seed"""
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False