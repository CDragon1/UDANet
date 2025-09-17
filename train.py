import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import numpy as np
from tqdm import tqdm
import logging
from datetime import datetime

# 导入项目模块
from models import build_model
from data import PestDataModule
from utils import (
    load_config, merge_config, setup_logger, 
    calculate_metrics, MetricTracker,
    save_checkpoint, load_checkpoint
)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Pest Detection and Counting Training')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='path to config file')
    parser.add_argument('--resume', type=str, default=None,
                        help='path to checkpoint to resume from')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='batch size for training')
    parser.add_argument('--num-epochs', type=int, default=None,
                        help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=None,
                        help='learning rate')
    parser.add_argument('--device', type=str, default='cuda',
                        help='device to use (cuda/cpu)')
    parser.add_argument('--save-dir', type=str, default='checkpoints',
                        help='directory to save checkpoints')
    parser.add_argument('--log-dir', type=str, default='logs',
                        help='directory to save logs')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed')
    
    return parser.parse_args()


def setup_training(args):
    """设置训练环境"""
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # 加载配置
    config = load_config(args.config)
    config = merge_config(config, args)
    
    # 创建输出目录
    os.makedirs(config.checkpoint.save_dir, exist_ok=True)
    os.makedirs(config.logging.log_dir, exist_ok=True)
    
    # 设置日志
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(config.logging.log_dir, f'training_{timestamp}.log')
    logger = setup_logger('PestTraining', log_file)
    
    # 设置设备
    device = torch.device(config.training.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    return config, logger, device


def build_loss_functions(config):
    """构建损失函数"""
    # 密度分级损失
    density_loss_fn = nn.CrossEntropyLoss()
    
    # 回归损失
    regression_loss_fn = nn.SmoothL1Loss()
    
    # 分类损失
    classification_loss_fn = nn.CrossEntropyLoss()
    
    # 密度图损失
    density_map_loss_fn = nn.MSELoss()
    
    return {
        'density': density_loss_fn,
        'regression': regression_loss_fn,
        'classification': classification_loss_fn,
        'density_map': density_map_loss_fn
    }


def train_one_epoch(model, dataloader, loss_fns, optimizer, device, epoch, config, logger, stage=1):
    """训练一个epoch"""
    model.train()
    metric_tracker = MetricTracker()
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{config.training.num_epochs} Stage {stage}')
    total_loss = 0
    
    for batch_idx, batch in enumerate(pbar):
        # 数据准备
        images = batch['image'].to(device)
        points_list = batch['points']
        density_maps_gt = batch['density_map'].to(device)
        density_labels = batch['density_level'].to(device)
        
        batch_size = images.size(0)
        
        optimizer.zero_grad()
        
        if stage == 1:
            # 第一阶段：密度分级训练
            outputs = model(images, training_stage=1)
            density_loss = loss_fns['density'](outputs['density_class'], density_labels)
            
            total_loss_batch = density_loss
            
            # 计算指标
            metrics = calculate_metrics(outputs, {'density_level': density_labels}, task_type='classification')
            
        else:
            # 第二阶段：联合训练
            outputs = model(images, training_stage=2)
            density_prediction = outputs['density_prediction']
            
            # 初始化损失
            regression_loss = classification_loss = density_map_loss = torch.tensor(0.0, device=device)
            
            # 低密度场景：点检测
            pd_mask = (density_prediction == 0)
            if pd_mask.any():
                # 这里需要实现点检测的损失计算
                # 简化处理：使用密度图损失作为替代
                regression_loss = loss_fns['density_map'](
                    outputs.get('pred_points', torch.zeros_like(density_maps_gt)),
                    density_maps_gt
                )
            
            # 高密度场景：密度图估计
            mdc_mask = (density_prediction == 1)
            if mdc_mask.any():
                density_map_loss = loss_fns['density_map'](
                    outputs.get('density_map', torch.zeros_like(density_maps_gt)),
                    density_maps_gt
                )
            
            # 密度分级损失
            density_loss = loss_fns['density'](outputs['density_class'], density_labels)
            
            # 总损失
            w_pd = config.training.loss_weights.pd_regression
            w_mdc = config.training.loss_weights.density_map
            w_density = config.training.loss_weights.density_classification
            
            total_loss_batch = (w_pd * regression_loss + 
                               w_mdc * density_map_loss + 
                               w_density * density_loss)
            
            # 计算指标
            metrics = calculate_metrics(outputs, {
                'density_level': density_labels,
                'density_map': density_maps_gt
            }, task_type='density')
        
        # 反向传播
        total_loss_batch.backward()
        optimizer.step()
        
        # 更新指标
        total_loss += total_loss_batch.item()
        metric_tracker.update(metrics, batch_size)
        
        # 更新进度条
        pbar.set_postfix({
            'Loss': f'{total_loss_batch.item():.4f}',
            'Avg_Loss': f'{total_loss / (batch_idx + 1):.4f}',
            **metric_tracker.get_metrics()
        })
    
    avg_loss = total_loss / len(dataloader)
    avg_metrics = metric_tracker.get_metrics()
    
    logger.info(f"Epoch {epoch+1} Stage {stage} - Loss: {avg_loss:.4f}, Metrics: {avg_metrics}")
    
    return avg_loss, avg_metrics


def validate(model, dataloader, loss_fns, device, epoch, config, logger, stage=1):
    """验证模型"""
    model.eval()
    metric_tracker = MetricTracker()
    
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f'Validation Epoch {epoch+1}'):
            # 数据准备
            images = batch['image'].to(device)
            points_list = batch['points']
            density_maps_gt = batch['density_map'].to(device)
            density_labels = batch['density_level'].to(device)
            
            if stage == 1:
                # 密度分级验证
                outputs = model(images, training_stage=1)
                density_loss = loss_fns['density'](outputs['density_class'], density_labels)
                
                total_loss += density_loss.item()
                
                # 计算指标
                metrics = calculate_metrics(outputs, {'density_level': density_labels}, task_type='classification')
                
            else:
                # 联合验证
                outputs = model(images, training_stage=2)
                density_prediction = outputs['density_prediction']
                
                # 计算密度图损失
                density_map_loss = loss_fns['density_map'](
                    outputs.get('density_map', torch.zeros_like(density_maps_gt)),
                    density_maps_gt
                )
                
                # 密度分级损失
                density_loss = loss_fns['density'](outputs['density_class'], density_labels)
                
                # 总损失
                total_loss += (density_map_loss + density_loss).item()
                
                # 计算指标
                metrics = calculate_metrics(outputs, {
                    'density_level': density_labels,
                    'density_map': density_maps_gt
                }, task_type='density')
            
            metric_tracker.update(metrics, images.size(0))
    
    avg_loss = total_loss / len(dataloader)
    avg_metrics = metric_tracker.get_metrics()
    
    logger.info(f"Validation Epoch {epoch+1} - Loss: {avg_loss:.4f}, Metrics: {avg_metrics}")
    
    return avg_loss, avg_metrics


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Unified Density-Aware Network (UDANet) Training')
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Configuration file path')
    parser.add_argument('--resume', type=str, default=None, help='Checkpoint path to resume training')
    parser.add_argument('--work_dir', type=str, default='work_dirs', help='Working directory')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Create working directory
    os.makedirs(args.work_dir, exist_ok=True)
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create logger
    logger = Logger(os.path.join(args.work_dir, 'logs'))
    logger.info("Starting training...")
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(config, args.work_dir)
    
    # Create model
    model = UnifiedNet(config)
    model.to(device)
    
    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    
    # Create loss function
    criterion = create_loss_function(config)
    
    # Create checkpoint manager
    checkpoint_manager = CheckpointManager(os.path.join(args.work_dir, 'checkpoints'))
    
    # Resume training
    start_epoch = 0
    if args.resume:
        checkpoint = checkpoint_manager.load_checkpoint(args.resume, model, optimizer)
        start_epoch = checkpoint['epoch'] + 1
    
    # Start training
    logger.info(f"Starting training from epoch {start_epoch}")
    
    for epoch in range(start_epoch, config.num_epochs):
        # Train one epoch
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, epoch, logger)
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device, epoch, logger)
        
        # Save checkpoint
        is_best = val_metrics.get('f1_score', 0) > 0.8  # Simple best model judgment
        checkpoint_manager.save_checkpoint(model, optimizer, epoch, val_metrics, is_best)
        
        # Log metrics
        logger.info(f"Epoch {epoch}: Training loss = {train_metrics['loss']:.4f}, "
                   f"Validation F1 = {val_metrics.get('f1_score', 0):.4f}")
    
    logger.info("Training completed!")


if __name__ == '__main__':
    main()
