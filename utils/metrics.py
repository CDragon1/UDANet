import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def MAE(pred, gt):
    """
    计算平均绝对误差 (Mean Absolute Error)
    
    Args:
        pred: 预测值
        gt: 真实值
        
    Returns:
        mae: 平均绝对误差
    """
    with torch.no_grad():
        pred = pred.view(-1)
        gt = gt.view(-1)
        mae = torch.abs(pred - gt).mean()
        return mae.item()


def RMSE(pred, gt):
    """
    计算均方根误差 (Root Mean Square Error)
    
    Args:
        pred: 预测值
        gt: 真实值
        
    Returns:
        rmse: 均方根误差
    """
    with torch.no_grad():
        pred = pred.view(-1)
        gt = gt.view(-1)
        mse = torch.mean((pred - gt) ** 2)
        rmse = torch.sqrt(mse)
        return rmse.item()


def Accuracy(pred, gt):
    """
    计算分类准确率
    
    Args:
        pred: 预测值
        gt: 真实值
        
    Returns:
        accuracy: 准确率
    """
    with torch.no_grad():
        if isinstance(pred, torch.Tensor):
            pred = pred.cpu().numpy()
        if isinstance(gt, torch.Tensor):
            gt = gt.cpu().numpy()
            
        pred_labels = np.argmax(pred, axis=1) if len(pred.shape) > 1 else pred
        accuracy = accuracy_score(gt, pred_labels)
        return accuracy


def Precision(pred, gt, average='macro'):
    """
    计算精确率
    
    Args:
        pred: 预测值
        gt: 真实值
        average: 平均方式
        
    Returns:
        precision: 精确率
    """
    with torch.no_grad():
        if isinstance(pred, torch.Tensor):
            pred = pred.cpu().numpy()
        if isinstance(gt, torch.Tensor):
            gt = gt.cpu().numpy()
            
        pred_labels = np.argmax(pred, axis=1) if len(pred.shape) > 1 else pred
        precision = precision_score(gt, pred_labels, average=average, zero_division=0)
        return precision


def Recall(pred, gt, average='macro'):
    """
    计算召回率
    
    Args:
        pred: 预测值
        gt: 真实值
        average: 平均方式
        
    Returns:
        recall: 召回率
    """
    with torch.no_grad():
        if isinstance(pred, torch.Tensor):
            pred = pred.cpu().numpy()
        if isinstance(gt, torch.Tensor):
            gt = gt.cpu().numpy()
            
        pred_labels = np.argmax(pred, axis=1) if len(pred.shape) > 1 else pred
        recall = recall_score(gt, pred_labels, average=average, zero_division=0)
        return recall


def F1Score(pred, gt, average='macro'):
    """
    计算F1分数
    
    Args:
        pred: 预测值
        gt: 真实值
        average: 平均方式
        
    Returns:
        f1: F1分数
    """
    with torch.no_grad():
        if isinstance(pred, torch.Tensor):
            pred = pred.cpu().numpy()
        if isinstance(gt, torch.Tensor):
            gt = gt.cpu().numpy()
            
        pred_labels = np.argmax(pred, axis=1) if len(pred.shape) > 1 else pred
        f1 = f1_score(gt, pred_labels, average=average, zero_division=0)
        return f1


def calculate_metrics(pred_points, gt_points, distance_threshold=10):
    """Calculate detection and counting metrics
    
    Args:
        pred_points: Predicted point coordinate list [(x1, y1), (x2, y2), ...]
        gt_points: Ground truth point coordinate list [(x1, y1), (x2, y2), ...]
        distance_threshold: Distance threshold for matching predicted and ground truth points
        
    Returns:
        dict: Dictionary containing various metrics
    """
    pred_count = len(pred_points)
    gt_count = len(gt_points)
    
    # If no ground truth points and no predicted points
    if pred_count == 0 and gt_count == 0:
        return {
            'mae': 0.0,
            'mse': 0.0,
            'rmse': 0.0,
            'precision': 1.0,
            'recall': 1.0,
            'f1_score': 1.0,
            'count_accuracy': 1.0
        }
    
    # Calculate counting error
    mae = abs(pred_count - gt_count)
    mse = (pred_count - gt_count) ** 2
    rmse = np.sqrt(mse)
    
    # Calculate counting accuracy
    count_accuracy = max(0, 1 - mae / max(pred_count, gt_count, 1))
    
    # If no predicted points or no ground truth points
    if pred_count == 0 or gt_count == 0:
        return {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'count_accuracy': count_accuracy
        }
    
    # Calculate detection metrics (based on distance matching)
    pred_points = np.array(pred_points)
    gt_points = np.array(gt_points)
    
    # Calculate distance matrix
    distances = cdist(pred_points, gt_points)
    
    # Use Hungarian algorithm for matching
    pred_matched, gt_matched = linear_sum_assignment(distances)
    
    # Calculate number of matches
    matches = 0
    for p_idx, g_idx in zip(pred_matched, gt_matched):
        if distances[p_idx, g_idx] <= distance_threshold:
            matches += 1
    
    # Calculate precision, recall and F1 score
    precision = matches / pred_count if pred_count > 0 else 0
    recall = matches / gt_count if gt_count > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'count_accuracy': count_accuracy
    }


class AverageMeter:
    """计算并存储平均值和当前值"""
    
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class MetricTracker:
    """指标追踪器"""
    
    def __init__(self):
        self.metrics = {}
        
    def update(self, metrics_dict, n=1):
        """更新指标"""
        for key, value in metrics_dict.items():
            if key not in self.metrics:
                self.metrics[key] = AverageMeter()
            self.metrics[key].update(value, n)
            
    def get_metrics(self):
        """获取当前指标"""
        return {key: meter.avg for key, meter in self.metrics.items()}
        
    def reset(self):
        """重置所有指标"""
        for meter in self.metrics.values():
            meter.reset()
            
    def __str__(self):
        """字符串表示"""
        metric_strs = []
        for key, meter in self.metrics.items():
            metric_strs.append(f"{key}: {meter.avg:.4f}")
        return ", ".join(metric_strs)


def count_parameters(model):
    """计算模型参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_summary(model, input_size=(3, 512, 512)):
    """模型摘要"""
    from torchsummary import summary
    
    device = next(model.parameters()).device
    summary(model, input_size=input_size, device=str(device))