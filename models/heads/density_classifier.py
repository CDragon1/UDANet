import torch
import torch.nn as nn
import torch.nn.functional as F


class DensityClassifier(nn.Module):
    """Density level classifier"""
    
    def __init__(self, input_dim, num_classes=2, hidden_dims=None):
        super(DensityClassifier, self).__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        
        if hidden_dims is None:
            hidden_dims = [512, 256]
        
        # Build classifier layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.classifier = nn.Sequential(*layers)
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input features [B, C] or [B, C, H, W]
            
        Returns:
            logits: Classification logits [B, num_classes]
        """
        # If input is feature map, apply global average pooling first
        if len(x.shape) == 4:
            x = F.adaptive_avg_pool2d(x, 1).squeeze(-1).squeeze(-1)
        
        # Classification
        logits = self.classifier(x)
        
        return logits


class DensityLevelLoss(nn.Module):
    """Density level classification loss function"""
    
    def __init__(self, class_weights=None, smoothing=0.0):
        super(DensityLevelLoss, self).__init__()
        
        self.class_weights = class_weights
        self.smoothing = smoothing
        
        if class_weights is not None:
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.criterion = nn.CrossEntropyLoss()
            
    def forward(self, pred_logits, gt_labels):
        """
        Calculate density level classification loss
        
        Args:
            pred_logits: Predicted logits [B, num_classes]
            gt_labels: Ground truth labels [B]
            
        Returns:
            loss: Loss value
        """
        loss = self.criterion(pred_logits, gt_labels)
        
        return loss


def calculate_density_level(num_points, threshold=10):
    """
    Calculate density level based on point count
    
    Args:
        num_points: Number of points
        threshold: Density threshold
        
    Returns:
        level: Density level (0: low density, 1: high density)
    """
    return 0 if num_points < threshold else 1


def adaptive_density_threshold(points_list, percentile=75):
    """
    Adaptively calculate density threshold
    
    Args:
        points_list: List of point lists
        percentile: Percentile value
        
    Returns:
        threshold: Adaptive threshold
    """
    if len(points_list) == 0:
        return 10
    
    point_counts = [len(points) for points in points_list]
    return np.percentile(point_counts, percentile)