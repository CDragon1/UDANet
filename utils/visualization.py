import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import cv2
from PIL import Image
import os
from typing import List, Dict, Optional, Tuple
import matplotlib.patches as patches


class Visualizer:
    """Visualization tool class"""
    
    def __init__(self, save_dir="visualizations"):
        """Initialize visualizer
        
        Args:
            save_dir: Save directory
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Set matplotlib style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def plot_image_with_points(self, image, points, save_name="image_with_points.png"):
        """Plot image with point annotations
        
        Args:
            image: Input image (H, W, 3) or (H, W)
            points: Point coordinate list [(x1, y1), (x2, y2), ...]
            save_name: Save file name
        """
        plt.figure(figsize=(12, 8))
        
        # Ensure image is in RGB format
        if len(image.shape) == 2:
            plt.imshow(image, cmap='gray')
        else:
            # Convert BGR to RGB (assuming input is BGR format)
            if image.shape[2] == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                plt.imshow(image_rgb)
            else:
                plt.imshow(image)
        
        # Plot points
        if points and len(points) > 0:
            points = np.array(points)
            plt.scatter(points[:, 0], points[:, 1], c='red', s=50, marker='o', alpha=0.7)
        
        plt.title(f'Image with Points (Count: {len(points) if points else 0})')
        plt.axis('off')
        
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
        
        return save_path
    
    def plot_density_map(self, density_map, save_name="density_map.png"):
        """Plot density map
        
        Args:
            density_map: Density map (H, W)
            save_name: Save file name
        """
        plt.figure(figsize=(10, 8))
        
        # Display density using heatmap
        plt.imshow(density_map, cmap='hot', interpolation='nearest')
        plt.colorbar(label='Density')
        plt.title('Density Map')
        plt.axis('off')
        
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
        
        return save_path
    
    def plot_comparison(self, image, points, density_map, save_name="comparison.png"):
        """Plot comparison (original image + point annotations vs density map)
        
        Args:
            image: Input image
            points: Point coordinates
            density_map: Density map
            save_name: Save file name
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        # Left: Original image + point annotations
        if len(image.shape) == 2:
            axes[0].imshow(image, cmap='gray')
        else:
            if image.shape[2] == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                axes[0].imshow(image_rgb)
            else:
                axes[0].imshow(image)
        
        if points and len(points) > 0:
            points = np.array(points)
            axes[0].scatter(points[:, 0], points[:, 1], c='red', s=50, marker='o', alpha=0.7)
        
        axes[0].set_title(f'Image with Points (Count: {len(points) if points else 0})')
        axes[0].axis('off')
        
        # Right: Density map
        im = axes[1].imshow(density_map, cmap='hot', interpolation='nearest')
        axes[1].set_title('Density Map')
        axes[1].axis('off')
        
        # Add colorbar
        plt.colorbar(im, ax=axes[1], label='Density')
        
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
        
        return save_path
    
    def visualize_batch_results(self, batch_data: Dict, 
                              predictions: Dict,
                              batch_idx: int,
                              save_dir: str = None):
        """可视化批次结果"""
        if save_dir is None:
            save_dir = self.save_dir
        
        images = batch_data['image'].cpu().numpy()
        density_maps_gt = batch_data['density_map'].cpu().numpy()
        density_levels_gt = batch_data['density_level'].cpu().numpy()
        
        pred_density_maps = predictions['density_map'].cpu().numpy()
        pred_density_levels = predictions['density_level'].cpu().numpy()
        
        batch_size = images.shape[0]
        
        for i in range(min(batch_size, 4)):  # 最多可视化4个样本
            # 处理图像
            img = images[i].transpose(1, 2, 0)
            img = (img - img.min()) / (img.max() - img.min())
            
            # 密度图
            gt_density = density_maps_gt[i].squeeze()
            pred_density = pred_density_maps[i].squeeze()
            
            # 数量
            gt_count = gt_density.sum()
            pred_count = pred_density.sum()
            
            # 密度等级
            gt_level = density_levels_gt[i]
            pred_level = pred_density_levels[i]
            
            save_path = os.path.join(save_dir, f'batch_{batch_idx}_sample_{i}.png')
            
            self.visualize_comparison(
                img, pred_density, gt_density, pred_count, gt_count,
                title=f"Sample {i} - GT Level: {gt_level}, Pred Level: {pred_level}",
                save_path=save_path
            )
    
    def plot_metrics_curves(self, metrics_history: Dict[str, List[float]],
                            save_path: Optional[str] = None):
        """绘制指标曲线"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 损失曲线
        if 'train_loss' in metrics_history and 'val_loss' in metrics_history:
            epochs = range(1, len(metrics_history['train_loss']) + 1)
            axes[0, 0].plot(epochs, metrics_history['train_loss'], 'b-', label='Train Loss')
            axes[0, 0].plot(epochs, metrics_history['val_loss'], 'r-', label='Val Loss')
            axes[0, 0].set_title('Loss Curves')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
        
        # MAE曲线
        if 'train_mae' in metrics_history and 'val_mae' in metrics_history:
            epochs = range(1, len(metrics_history['train_mae']) + 1)
            axes[0, 1].plot(epochs, metrics_history['train_mae'], 'b-', label='Train MAE')
            axes[0, 1].plot(epochs, metrics_history['val_mae'], 'r-', label='Val MAE')
            axes[0, 1].set_title('MAE Curves')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('MAE')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
        
        # RMSE曲线
        if 'train_rmse' in metrics_history and 'val_rmse' in metrics_history:
            epochs = range(1, len(metrics_history['train_rmse']) + 1)
            axes[1, 0].plot(epochs, metrics_history['train_rmse'], 'b-', label='Train RMSE')
            axes[1, 0].plot(epochs, metrics_history['val_rmse'], 'r-', label='Val RMSE')
            axes[1, 0].set_title('RMSE Curves')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('RMSE')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        
        # 密度等级准确率
        if 'density_acc' in metrics_history:
            epochs = range(1, len(metrics_history['density_acc']) + 1)
            axes[1, 1].plot(epochs, metrics_history['density_acc'], 'g-', label='Density Acc')
            axes[1, 1].set_title('Density Level Accuracy')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Accuracy')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.close()
        return fig
    
    def plot_error_distribution(self, errors: List[float],
                              title: str = "Error Distribution",
                              save_path: Optional[str] = None):
        """绘制误差分布"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # 直方图
        axes[0].hist(errors, bins=30, alpha=0.7, color='blue', edgecolor='black')
        axes[0].set_title(f'{title} - Histogram')
        axes[0].set_xlabel('Error')
        axes[0].set_ylabel('Frequency')
        axes[0].grid(True, alpha=0.3)
        
        # 箱线图
        axes[1].boxplot(errors)
        axes[1].set_title(f'{title} - Box Plot')
        axes[1].set_ylabel('Error')
        axes[1].grid(True, alpha=0.3)
        
        plt.suptitle(f'{title}\nMean: {np.mean(errors):.3f}, Std: {np.std(errors):.3f}')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.close()
        return fig
    
    def plot_scatter_comparison(self, pred_counts: List[float],
                              gt_counts: List[float],
                              title: str = "Count Comparison",
                              save_path: Optional[str] = None):
        """绘制散点对比图"""
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        # 散点图
        ax.scatter(gt_counts, pred_counts, alpha=0.6, s=50)
        
        # 完美预测线
        min_val = min(min(gt_counts), min(pred_counts))
        max_val = max(max(gt_counts), max(pred_counts))
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
        
        # 计算R²
        correlation = np.corrcoef(gt_counts, pred_counts)[0, 1]
        
        ax.set_xlabel('Ground Truth Count')
        ax.set_ylabel('Predicted Count')
        ax.set_title(f'{title}\nCorrelation: {correlation:.3f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.close()
        return fig
    
    def create_summary_figure(self, results: Dict,
                            save_path: Optional[str] = None):
        """创建结果摘要图"""
        fig = plt.figure(figsize=(20, 16))
        
        # 创建子图网格
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
        
        # 1. 样本展示
        if 'sample_images' in results:
            for i, (img, pred, gt) in enumerate(results['sample_images'][:4]):
                ax = fig.add_subplot(gs[0, i])
                ax.imshow(img)
                ax.set_title(f'Sample {i+1}\nPred: {pred:.1f}, GT: {gt:.1f}')
                ax.axis('off')
        
        # 2. 指标表格
        ax_table = fig.add_subplot(gs[1, :2])
        ax_table.axis('off')
        
        if 'metrics' in results:
            metrics_data = []
            for metric, value in results['metrics'].items():
                metrics_data.append([metric, f"{value:.4f}"])
            
            table = ax_table.table(cellText=metrics_data,
                                   colLabels=['Metric', 'Value'],
                                   cellLoc='center',
                                   loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(12)
            table.scale(1.2, 1.5)
            ax_table.set_title('Test Metrics', fontsize=14, pad=20)
        
        # 3. 误差分布
        if 'errors' in results:
            ax_error = fig.add_subplot(gs[1, 2:])
            ax_error.hist(results['errors'], bins=30, alpha=0.7, color='blue')
            ax_error.set_title('Error Distribution')
            ax_error.set_xlabel('Error')
            ax_error.set_ylabel('Frequency')
            ax_error.grid(True, alpha=0.3)
        
        # 4. 散点图
        if 'pred_counts' in results and 'gt_counts' in results:
            ax_scatter = fig.add_subplot(gs[2, :2])
            ax_scatter.scatter(results['gt_counts'], results['pred_counts'], alpha=0.6)
            
            # 完美预测线
            min_val = min(min(results['gt_counts']), min(results['pred_counts']))
            max_val = max(max(results['gt_counts']), max(results['pred_counts']))
            ax_scatter.plot([min_val, max_val], [min_val, max_val], 'r--')
            
            ax_scatter.set_xlabel('Ground Truth Count')
            ax_scatter.set_ylabel('Predicted Count')
            ax_scatter.set_title('Count Comparison')
            ax_scatter.grid(True, alpha=0.3)
        
        # 5. 密度等级分布
        if 'density_levels' in results:
            ax_density = fig.add_subplot(gs[2, 2:])
            density_counts = {}
            for level in results['density_levels']:
                density_counts[level] = density_counts.get(level, 0) + 1
            
            ax_density.bar(density_counts.keys(), density_counts.values())
            ax_density.set_xlabel('Density Level')
            ax_density.set_ylabel('Count')
            ax_density.set_title('Density Level Distribution')
            ax_density.set_xticks(list(density_counts.keys()))
        
        # 6. 训练曲线（如果有）
        if 'training_history' in results:
            ax_train = fig.add_subplot(gs[3, :])
            history = results['training_history']
            
            if 'train_loss' in history:
                epochs = range(1, len(history['train_loss']) + 1)
                ax_train.plot(epochs, history['train_loss'], label='Train Loss')
                if 'val_loss' in history:
                    ax_train.plot(epochs, history['val_loss'], label='Val Loss')
                ax_train.set_xlabel('Epoch')
                ax_train.set_ylabel('Loss')
                ax_train.set_title('Training History')
                ax_train.legend()
                ax_train.grid(True, alpha=0.3)
        
        plt.suptitle('Pest Detection and Counting Results Summary', fontsize=16)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.close()
        return fig


def save_visualization_results(visualizations: Dict, output_dir: str):
    """保存可视化结果"""
    os.makedirs(output_dir, exist_ok=True)
    
    for name, fig in visualizations.items():
        save_path = os.path.join(output_dir, f'{name}.png')
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    print(f"Visualization results saved to {output_dir}")