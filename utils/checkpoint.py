import os
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List
import json
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Checkpoint manager"""
    
    def __init__(self, checkpoint_dir: str = "checkpoints", 
                 max_checkpoints: int = 5,
                 save_best_only: bool = True,
                 monitor_metric: str = "val_loss",
                 mode: str = "min"):
        """
        Args:
            checkpoint_dir: Checkpoint save directory
            max_checkpoints: Maximum number of checkpoints to keep
            save_best_only: Whether to save only the best model
            monitor_metric: Metric to monitor
            mode: Metric mode ('min' or 'max')
        """
        self.checkpoint_dir = checkpoint_dir
        self.max_checkpoints = max_checkpoints
        self.save_best_only = save_best_only
        self.monitor_metric = monitor_metric
        self.mode = mode
        
        # Ensure checkpoint directory exists
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Best metric value
        self.best_metric_value = float('inf') if mode == 'min' else float('-inf')
        
        # List of saved checkpoints
        self.saved_checkpoints = []
        
        # Metadata file path
        self.metadata_file = os.path.join(checkpoint_dir, "checkpoint_metadata.json")
        
        # Load existing metadata
        self.load_metadata()
    
    def save_checkpoint(self, model: nn.Module, optimizer: torch.optim.Optimizer,
                       epoch: int, metrics: Dict[str, float],
                       config: Dict[str, Any] = None,
                       is_best: bool = False) -> str:
        """
        保存检查点
        
        Args:
            model: 模型
            optimizer: 优化器
            epoch: 当前epoch
            metrics: 指标字典
            config: 配置信息
            is_best: 是否为最佳模型
        
        Returns:
            保存的文件路径
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'config': config,
            'timestamp': datetime.now().isoformat(),
            'model_architecture': model.__class__.__name__
        }
        
        # 生成检查点文件名
        checkpoint_name = f"checkpoint_epoch_{epoch:03d}"
        if is_best:
            checkpoint_name += "_best"
        checkpoint_name += f"_{self.monitor_metric}_{metrics.get(self.monitor_metric, 0):.4f}.pth"
        
        checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_name)
        
        # 保存检查点
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")
        
        # 更新最佳指标
        current_metric = metrics.get(self.monitor_metric, 0)
        if self._is_better_metric(current_metric):
            self.best_metric_value = current_metric
            is_best = True
        
        # 保存最佳模型
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, "best_model.pth")
            torch.save(checkpoint, best_path)
            logger.info(f"Best model saved: {best_path}")
        
        # 更新检查点列表
        self.saved_checkpoints.append({
            'path': checkpoint_path,
            'epoch': epoch,
            'metrics': metrics,
            'is_best': is_best,
            'timestamp': datetime.now().isoformat()
        })
        
        # 清理旧检查点
        self._cleanup_checkpoints()
        
        # 保存元数据
        self.save_metadata()
        
        return checkpoint_path
    
    def load_checkpoint(self, checkpoint_path: str, model: nn.Module, 
                       optimizer: torch.optim.Optimizer = None,
                       device: str = 'cpu') -> Dict[str, Any]:
        """
        加载检查点
        
        Args:
            checkpoint_path: 检查点文件路径
            model: 模型
            optimizer: 优化器（可选）
            device: 设备
        
        Returns:
            检查点信息
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # 加载模型状态
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # 加载优化器状态
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        logger.info(f"Checkpoint loaded: {checkpoint_path}")
        logger.info(f"Epoch: {checkpoint.get('epoch', 'Unknown')}")
        logger.info(f"Metrics: {checkpoint.get('metrics', {})}")
        
        return checkpoint
    
    def load_best_checkpoint(self, model: nn.Module, 
                            optimizer: torch.optim.Optimizer = None,
                            device: str = 'cpu') -> Optional[Dict[str, Any]]:
        """加载最佳检查点"""
        best_path = os.path.join(self.checkpoint_dir, "best_model.pth")
        
        if os.path.exists(best_path):
            return self.load_checkpoint(best_path, model, optimizer, device)
        else:
            logger.warning("Best checkpoint not found")
            return None
    
    def load_latest_checkpoint(self, model: nn.Module, 
                              optimizer: torch.optim.Optimizer = None,
                              device: str = 'cpu') -> Optional[Dict[str, Any]]:
        """加载最新的检查点"""
        if not self.saved_checkpoints:
            logger.warning("No checkpoints found")
            return None
        
        # 按时间排序，获取最新的检查点
        latest_checkpoint = max(self.saved_checkpoints, 
                               key=lambda x: x['timestamp'])
        
        return self.load_checkpoint(latest_checkpoint['path'], model, optimizer, device)
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """列出所有检查点"""
        return self.saved_checkpoints.copy()
    
    def get_best_checkpoint_path(self) -> Optional[str]:
        """获取最佳检查点路径"""
        best_path = os.path.join(self.checkpoint_dir, "best_model.pth")
        return best_path if os.path.exists(best_path) else None
    
    def _is_better_metric(self, current_metric: float) -> bool:
        """判断当前指标是否更好"""
        if self.mode == 'min':
            return current_metric < self.best_metric_value
        else:
            return current_metric > self.best_metric_value
    
    def _cleanup_checkpoints(self):
        """清理旧的检查点"""
        if len(self.saved_checkpoints) <= self.max_checkpoints:
            return
        
        # 按时间排序
        self.saved_checkpoints.sort(key=lambda x: x['timestamp'])
        
        # 保留最佳模型和最新的检查点
        best_checkpoints = [cp for cp in self.saved_checkpoints if cp['is_best']]
        regular_checkpoints = [cp for cp in self.saved_checkpoints if not cp['is_best']]
        
        # 删除旧的常规检查点
        while len(regular_checkpoints) > self.max_checkpoints - len(best_checkpoints):
            old_checkpoint = regular_checkpoints.pop(0)
            if os.path.exists(old_checkpoint['path']):
                os.remove(old_checkpoint['path'])
                logger.info(f"Removed old checkpoint: {old_checkpoint['path']}")
        
        # 更新检查点列表
        self.saved_checkpoints = best_checkpoints + regular_checkpoints
    
    def save_metadata(self):
        """保存元数据"""
        metadata = {
            'monitor_metric': self.monitor_metric,
            'mode': self.mode,
            'best_metric_value': self.best_metric_value,
            'saved_checkpoints': self.saved_checkpoints,
            'last_updated': datetime.now().isoformat()
        }
        
        with open(self.metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
    
    def load_metadata(self):
        """加载元数据"""
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                
                self.best_metric_value = metadata.get('best_metric_value', 
                    float('inf') if self.mode == 'min' else float('-inf'))
                self.saved_checkpoints = metadata.get('saved_checkpoints', [])
                
                logger.info("Loaded checkpoint metadata")
            except Exception as e:
                logger.warning(f"Failed to load metadata: {e}")


class ModelManager:
    """模型管理器"""
    
    def __init__(self, checkpoint_dir: str = "checkpoints"):
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_manager = CheckpointManager(checkpoint_dir)
    
    def save_model(self, model: nn.Module, optimizer: torch.optim.Optimizer,
                  epoch: int, metrics: Dict[str, float], 
                  config: Dict[str, Any] = None) -> str:
        """保存模型"""
        return self.checkpoint_manager.save_checkpoint(
            model, optimizer, epoch, metrics, config
        )
    
    def load_model(self, model_path: str, model: nn.Module,
                  optimizer: torch.optim.Optimizer = None,
                  device: str = 'cpu') -> Dict[str, Any]:
        """加载模型"""
        return self.checkpoint_manager.load_checkpoint(
            model_path, model, optimizer, device
        )
    
    def load_best_model(self, model: nn.Module,
                       optimizer: torch.optim.Optimizer = None,
                       device: str = 'cpu') -> Optional[Dict[str, Any]]:
        """加载最佳模型"""
        return self.checkpoint_manager.load_best_checkpoint(
            model, optimizer, device
        )
    
    def export_model(self, model: nn.Module, export_path: str,
                    input_size: tuple = (1, 3, 224, 224),
                    device: str = 'cpu'):
        """导出模型为ONNX格式"""
        try:
            import torch.onnx
            
            model.eval()
            model = model.to(device)
            
            # 创建示例输入
            dummy_input = torch.randn(input_size).to(device)
            
            # 导出ONNX模型
            torch.onnx.export(
                model,
                dummy_input,
                export_path,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                }
            )
            
            logger.info(f"Model exported to ONNX: {export_path}")
            
        except ImportError:
            logger.error("ONNX export requires torch.onnx module")
        except Exception as e:
            logger.error(f"Failed to export model: {e}")
    
    def create_model_summary(self, model: nn.Module, 
                           input_size: tuple = (1, 3, 224, 224)) -> Dict[str, Any]:
        """创建模型摘要"""
        summary = {
            'model_type': model.__class__.__name__,
            'total_parameters': sum(p.numel() for p in model.parameters()),
            'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
            'input_size': input_size,
            'architecture': str(model)
        }
        
        return summary


def create_checkpoint_manager(checkpoint_dir: str = "checkpoints", **kwargs) -> CheckpointManager:
    """创建检查点管理器"""
    return CheckpointManager(checkpoint_dir, **kwargs)


def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer,
                   epoch: int, metrics: Dict[str, float],
                   checkpoint_path: str, config: Dict[str, Any] = None):
    """保存检查点（简化版）"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'config': config,
        'timestamp': datetime.now().isoformat()
    }
    
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Checkpoint saved: {checkpoint_path}")


def load_checkpoint(checkpoint_path: str, model: nn.Module,
                   optimizer: torch.optim.Optimizer = None,
                   device: str = 'cpu') -> Dict[str, Any]:
    """加载检查点（简化版）"""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    logger.info(f"Checkpoint loaded: {checkpoint_path}")
    return checkpoint