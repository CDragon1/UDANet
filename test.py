import os
import argparse
import torch
import numpy as np
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# 导入项目模块
from models import build_model
from data import PestDataModule
from utils import (
    load_config, merge_config, setup_logger,
    calculate_metrics, MetricTracker,
    load_checkpoint, save_results
)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Pest Detection and Counting Testing')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='path to config file')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='path to checkpoint file')
    parser.add_argument('--output-dir', type=str, default='results',
                        help='directory to save results')
    parser.add_argument('--device', type=str, default='cuda',
                        help='device to use (cuda/cpu)')
    parser.add_argument('--save-visualizations', action='store_true',
                        help='save visualization results')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='batch size for testing')
    
    return parser.parse_args()


def setup_testing(args):
    """设置测试环境"""
    # 加载配置
    config = load_config(args.config)
    config = merge_config(config, args)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'visualizations'), exist_ok=True)
    
    # 设置日志
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(args.output_dir, f'testing_{timestamp}.log')
    logger = setup_logger('PestTesting', log_file)
    
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    return config, logger, device


def visualize_results(images, predictions, ground_truths, save_dir, batch_idx):
    """可视化测试结果"""
    batch_size = images.size(0)
    
    for i in range(min(batch_size, 4)):  # 每个批次最多可视化4张图
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 原始图像
        img = images[i].cpu().numpy().transpose(1, 2, 0)
        img = (img - img.min()) / (img.max() - img.min())  # 归一化
        axes[0, 0].imshow(img)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # 预测密度图
        pred_density = predictions['density_map'][i].cpu().numpy().squeeze()
        im1 = axes[0, 1].imshow(pred_density, cmap='hot')
        axes[0, 1].set_title(f'Predicted Density Map (Count: {pred_density.sum():.1f})')
        axes[0, 1].axis('off')
        plt.colorbar(im1, ax=axes[0, 1])
        
        # 真实密度图
        gt_density = ground_truths['density_map'][i].cpu().numpy().squeeze()
        im2 = axes[1, 0].imshow(gt_density, cmap='hot')
        axes[1, 0].set_title(f'Ground Truth Density Map (Count: {gt_density.sum():.1f})')
        axes[1, 0].axis('off')
        plt.colorbar(im2, ax=axes[1, 0])
        
        # 密度等级预测
        pred_level = predictions['density_level'][i].item()
        gt_level = ground_truths['density_level'][i].item()
        axes[1, 1].text(0.5, 0.7, f'Predicted Level: {pred_level}', 
                         ha='center', va='center', fontsize=14, 
                         color='green' if pred_level == gt_level else 'red')
        axes[1, 1].text(0.5, 0.3, f'Ground Truth Level: {gt_level}', 
                         ha='center', va='center', fontsize=14, color='blue')
        axes[1, 1].set_title('Density Level Classification')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        save_path = os.path.join(save_dir, f'results_batch_{batch_idx}_sample_{i}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()


def test_model(model, dataloader, device, config, logger, args):
    """测试模型"""
    model.eval()
    metric_tracker = MetricTracker()
    
    all_predictions = []
    all_ground_truths = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Testing')
        
        for batch_idx, batch in enumerate(pbar):
            # 数据准备
            images = batch['image'].to(device)
            points_list = batch['points']
            density_maps_gt = batch['density_map'].to(device)
            density_labels = batch['density_level'].to(device)
            
            # 前向传播
            outputs = model(images, training_stage=2)
            
            # 准备预测和真实值
            predictions = {
                'density_map': outputs.get('density_map', torch.zeros_like(density_maps_gt)),
                'density_level': outputs['density_class'].argmax(dim=1),
                'density_prediction': outputs.get('density_prediction', torch.zeros(images.size(0), device=device))
            }
            
            ground_truths = {
                'density_map': density_maps_gt,
                'density_level': density_labels,
                'points': points_list
            }
            
            # 计算指标
            metrics = calculate_metrics(predictions, ground_truths, task_type='density')
            metric_tracker.update(metrics, images.size(0))
            
            # 保存预测结果
            all_predictions.append({k: v.cpu() for k, v in predictions.items()})
            all_ground_truths.append({k: v.cpu() if torch.is_tensor(v) else v for k, v in ground_truths.items()})
            
            # 更新进度条
            pbar.set_postfix(metric_tracker.get_metrics())
            
            # 可视化（如果启用）
            if args.save_visualizations and batch_idx < 10:  # 只可视化前10个批次
                visualize_results(images, predictions, ground_truths, 
                                os.path.join(args.output_dir, 'visualizations'), batch_idx)
    
    # 计算最终指标
    final_metrics = metric_tracker.get_metrics()
    logger.info(f"Test Results: {final_metrics}")
    
    return final_metrics, all_predictions, all_ground_truths


def generate_report(metrics, config, args, logger):
    """生成测试报告"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = os.path.join(args.output_dir, f'test_report_{timestamp}.json')
    
    report = {
        'timestamp': timestamp,
        'config_file': args.config,
        'checkpoint_file': args.checkpoint,
        'model_config': {
            'backbone': config.model.backbone,
            'num_classes': config.model.num_classes,
            'input_size': config.data.input_size
        },
        'test_metrics': metrics,
        'test_settings': {
            'batch_size': args.batch_size,
            'device': args.device,
            'save_visualizations': args.save_visualizations
        }
    }
    
    # 保存报告
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Test report saved to {report_file}")
    
    # 生成摘要报告
    summary_file = os.path.join(args.output_dir, 'test_summary.txt')
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("Pest Detection and Counting Test Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Test Date: {timestamp}\n")
        f.write(f"Model Configuration: {config.model.backbone}\n")
        f.write(f"Checkpoint: {args.checkpoint}\n\n")
        f.write("Test Metrics:\n")
        for metric_name, metric_value in metrics.items():
            f.write(f"  {metric_name}: {metric_value:.4f}\n")
    
    logger.info(f"Test summary saved to {summary_file}")
    
    return report_file


def main():
    """主测试函数"""
    # 解析参数
    args = parse_args()
    
    # 设置测试环境
    config, logger, device = setup_testing(args)
    
    # 构建模型
    model = build_model(config).to(device)
    logger.info(f"Model built with {sum(p.numel() for p in model.parameters())} parameters")
    
    # 加载检查点
    load_checkpoint(args.checkpoint, model, None, logger)
    
    # 数据模块
    data_module = PestDataModule(config)
    data_module.setup()
    test_loader = data_module.test_dataloader()
    
    logger.info(f"Testing on {len(data_module.test_dataset)} samples")
    
    # 测试模型
    metrics, predictions, ground_truths = test_model(model, test_loader, device, config, logger, args)
    
    # 生成报告
    report_file = generate_report(metrics, config, args, logger)
    
    # 保存预测结果
    results_file = os.path.join(args.output_dir, 'test_results.pth')
    torch.save({
        'predictions': predictions,
        'ground_truths': ground_truths,
        'metrics': metrics,
        'config': config,
        'args': args
    }, results_file)
    
    logger.info(f"Test results saved to {results_file}")
    logger.info("Testing completed!")
    
    # 打印最终结果
    print("\n" + "="*50)
    print("TEST RESULTS SUMMARY")
    print("="*50)
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")
    print(f"\nDetailed results saved to: {args.output_dir}")


if __name__ == '__main__':
    main()