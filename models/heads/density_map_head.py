import torch
import torch.nn as nn
import torch.nn.functional as F


class SCAModule(nn.Module):
    """Spatial and Channel Attention Module"""
    
    def __init__(self, in_channels, num_classes):
        super(SCAModule, self).__init__()
        
        # Spatial attention
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(channels, channels // 8, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 8, 1, 1),
            nn.Sigmoid()
        )
        
        # Channel attention
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 8, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 8, channels, 1),
            nn.Sigmoid()
        )
        
        # Final output layer
        self.output_conv = nn.Conv2d(channels, num_classes, 1)

    def forward(self, x):
        """Forward pass
        
        Args:
            x: Input feature map [B, C, H, W]
            
        Returns:
            output: Output density map [B, num_classes, H, W]
        """
        # Spatial attention
        spatial_att = self.spatial_attention(x)
        x_spatial = x * spatial_att
        
        # Channel attention
        channel_att = self.channel_attention(x)
        x_channel = x * channel_att
        
        # Fuse features
        x_fused = x_spatial + x_channel + x
        
        # Final output
        output = self.output_conv(x_fused)
        return output


class DensityMapHead(nn.Module):
    """Density map prediction head"""
    
    def __init__(self, input_channels, unified_channels=64):
        super(DensityMapHead, self).__init__()
        
        # FPN feature fusion
        self.fpn_conv = nn.ModuleList([
            nn.Conv2d(input_channels[i], unified_channels, 1) 
            for i in range(len(input_channels))
        ])
        
        # Feature fusion
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(unified_channels * len(input_channels), unified_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(unified_channels, unified_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Upsampling layer
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        
        # Refinement layer
        self.refine_conv = nn.Sequential(
            nn.Conv2d(unified_channels, unified_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(unified_channels, unified_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Attention module
        self.attention = SCAModule(unified_channels, 1)
        
    def forward(self, fpn_features):
        """Forward pass
        
        Args:
            fpn_features: FPN multi-scale features list [P3, P4, P5]
            
        Returns:
            density_map: Density map [B, 1, H, W]
        """
        # Process different scale features
        processed_features = []
        target_size = None
        
        for i, feat in enumerate(fpn_features):
            # Unify channels
            processed_feat = self.fpn_conv[i](feat)
            processed_features.append(processed_feat)
            
            # Record target size (use middle layer)
            if i == len(fpn_features) // 2:
                target_size = processed_feat.shape[2:]
        
        # Unify sizes
        unified_features = []
        for feat in processed_features:
            if feat.shape[2:] != target_size:
                feat = F.interpolate(feat, size=target_size, mode='bilinear', align_corners=False)
            unified_features.append(feat)
        
        # Feature fusion
        fused_feature = torch.cat(unified_features, dim=1)
        fused_feature = self.fusion_conv(fused_feature)
        
        # Upsampling
        upsampled_feature = self.upsample(fused_feature)
        
        # Refinement
        refined_feature = self.refine_conv(upsampled_feature)
        
        # Attention mechanism and final output
        density_map = self.attention(refined_feature)
        
        return density_map


class DensityMapLoss(nn.Module):
    """Density map loss function"""
    
    def __init__(self, loss_type='mse', reduction='mean'):
        super(DensityMapLoss, self).__init__()
        self.loss_type = loss_type
        self.reduction = reduction
        
        if loss_type == 'mse':
            self.loss_fn = nn.MSELoss(reduction=reduction)
        elif loss_type == 'l1':
            self.loss_fn = nn.L1Loss(reduction=reduction)
        elif loss_type == 'smooth_l1':
            self.loss_fn = nn.SmoothL1Loss(reduction=reduction)
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")
    
    def forward(self, pred_density, gt_density, mask=None):
        """Calculate density map loss
        
        Args:
            pred_density: Predicted density map [B, 1, H, W]
            gt_density: Ground truth density map [B, 1, H, W]
            mask: Optional mask [B, 1, H, W]
            
        Returns:
            loss: Loss value
        """
        if mask is not None:
            pred_density = pred_density * mask
            gt_density = gt_density * mask
        
        return self.loss_fn(pred_density, gt_density)


def generate_density_map(points, image_shape, sigma=4.0):
    """Generate density map
    
    Args:
        points: Point coordinates list [[x1, y1], [x2, y2], ...]
        image_shape: Image size (H, W)
        sigma: Gaussian kernel standard deviation
        
    Returns:
        density_map: Density map [H, W]
    """
    h, w = image_shape
    density_map = np.zeros((h, w), dtype=np.float32)
    
    if len(points) == 0:
        return density_map
    
    # Generate Gaussian distribution for each point
    for point in points:
        x, y = int(point[0]), int(point[1])
        if 0 <= x < w and 0 <= y < h:
            # Generate Gaussian distribution centered at this point
            gaussian = np.zeros((h, w), dtype=np.float32)
            gaussian[y, x] = 1.0
            
            # Apply Gaussian filter
            density_map += gaussian_filter(gaussian, sigma=sigma)
    
    return density_map