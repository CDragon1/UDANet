import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbones import SwinTransformer
from .heads import DetectionHead, DensityMapHead, DensityClassifier
from .necks import FPN


class UnifiedNet(nn.Module):
    """
    Unified pest detection and counting network
    Supports adaptive detection in multi-density scenarios
    """
    
    def __init__(self, backbone='swin', num_classes=7, row=2, line=2, pretrained=True):
        super(UnifiedNet, self).__init__()
        self.num_classes = num_classes
        self.row = row
        self.line = line
        
        # Backbone network
        if backbone == 'swin':
            self.backbone = SwinTransformer(pretrained=pretrained)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Density level classification module
        self.density_classifier = DensityClassifier(
            input_dim=self.backbone.out_dim,
            num_classes=2
        )
        
        # Detection head (for low-density scenarios)
        self.detection_head = DetectionHead(
            input_dim=self.backbone.out_dim,
            num_classes=num_classes,
            num_anchor_points=row * line
        )
        
        # Density map head (for high-density scenarios)
        self.density_map_head = DensityMapHead(
            input_channels=self.backbone.out_channels,
            unified_channels=64
        )
        
        # FPN neck network
        self.neck = FPN(
            in_channels=self.backbone.out_channels,
            out_channels=128,
            num_levels=len(self.backbone.out_channels)
        )
        
    def forward(self, rgb, points=None, training_stage=None):
        """Forward pass
        
        Args:
            rgb: Input image [B, 3, H, W]
            points: Ground truth point coordinates (used during training)
            training_stage: Training stage (1: density classification, 2: joint training)
            
        Returns:
            dict: Dictionary containing various outputs
        """
        if training_stage is None:
            training_stage = 2  # default joint training
            
        # Extract multi-scale features
        features = self.backbone(rgb)
        
        # Density level classification
        density_features = features[-1]  # Use highest-level features
        density_logits = self.density_classifier(density_features)
        density_class = F.softmax(density_logits, dim=1)
        
        outputs = {
            'density_logits': density_logits,
            'density_class': density_class
        }
        
        if training_stage == 1:
            # Stage 1: Density classification only
            return outputs
        
        elif training_stage == 2:
            # Stage 2: Joint training
            density_prediction = density_class.argmax(dim=1)  # [B], 0: low density, 1: high density
            
            # FPN feature fusion
            fpn_features = self.neck(features)
            
            # Choose different heads based on density prediction
            for i in range(rgb.shape[0]):
                if density_prediction[i] == 0:  # Low density: use point detection
                    detection_out = self.detection_head(fpn_features[1][i:i+1])  # Use middle-level features
                    outputs.update({
                        'detection_regression': detection_out['regression'],
                        'detection_classification': detection_out['classification'],
                        'anchor_points': detection_out['anchor_points']
                    })
                else:  # High density: use density map estimation
                    density_map = self.density_map_head([f[i:i+1] for f in fpn_features])
                    outputs['density_map'] = density_map
            
            return outputs
        
        else:
            raise ValueError(f"Invalid training_stage: {training_stage}")
    
    def calculate_density_level(self, points, threshold=None):
        """Calculate density level based on point count
        
        Args:
            points: Point coordinate list
            threshold: Density threshold
            
        Returns:
            int: Density level (0: low density, 1: high density)
        """
        if threshold is None:
            threshold = 10  # default threshold
        
        num_points = len(points) if points is not None else 0
        return 0 if num_points < threshold else 1


def build_model(config):
    """
    根据配置构建模型
    
    Args:
        config: 配置对象
        
    Returns:
        UnifiedNet: 构建好的模型
    """
    model = UnifiedNet(
        backbone=config.model.backbone,
        num_classes=config.model.num_classes,
        row=config.model.detection.row,
        line=config.model.detection.line,
        pretrained=config.model.pretrained
    )
    return model


if __name__ == '__main__':
    # 测试模型
    model = UnifiedNet(num_classes=7)
    x = torch.randn(2, 3, 512, 512)
    
    # 测试第一阶段
    out1 = model(x, training_stage=1)
    print("Stage 1 output:", out1.keys())
    
    # 测试第二阶段
    out2 = model(x, training_stage=2)
    print("Stage 2 output:", out2.keys())
    
    print("Model created successfully!")