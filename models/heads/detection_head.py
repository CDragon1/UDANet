import torch
import torch.nn as nn
import torch.nn.functional as F


class AnchorPoints(nn.Module):
    """Generate anchor point coordinates"""
    
    def __init__(self, pyramid_levels=[3], row=2, line=2):
        super(AnchorPoints, self).__init__()
        self.pyramid_levels = pyramid_levels
        self.row = row
        self.line = line
        
    def forward(self, img_batch):
        """
        Generate anchor point coordinates
        
        Args:
            img_batch: input image batch [B, C, H, W]
            
        Returns:
            anchor_points: anchor point coordinates [B, num_anchors, 2]
        """
        batch_size = img_batch.shape[0]
        device = img_batch.device
        
        # compute anchor grid
        anchor_points = []
        for level in self.pyramid_levels:
            stride = 2 ** level
            feat_h = img_batch.shape[2] // stride
            feat_w = img_batch.shape[3] // stride
            
            # generate grid coordinates
            shift_x = torch.arange(0, feat_w, device=device) * stride + stride // 2
            shift_y = torch.arange(0, feat_h, device=device) * stride + stride // 2
            shift_yy, shift_xx = torch.meshgrid(shift_y, shift_x)
            
            # flatten coordinates
            shift_xx = shift_xx.reshape(-1)
            shift_yy = shift_yy.reshape(-1)
            
            # generate anchors
            for i in range(self.row):
                for j in range(self.line):
                    offset_x = (j + 0.5) * stride / self.line
                    offset_y = (i + 0.5) * stride / self.row
                    
                    points_x = shift_xx + offset_x
                    points_y = shift_yy + offset_y
                    
                    points = torch.stack([points_x, points_y], dim=1)
                    anchor_points.append(points)
        
        # concatenate all anchors
        anchor_points = torch.cat(anchor_points, dim=0)
        anchor_points = anchor_points.unsqueeze(0).repeat(batch_size, 1, 1)
        
        return anchor_points


class RegressionModel(nn.Module):
    """Regression model for predicting point coordinate offsets"""
    
    def __init__(self, num_features_in, num_anchor_points):
        super(RegressionModel, self).__init__()
        
        self.conv1 = nn.Conv2d(num_features_in, 256, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()
        
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()
        
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()
        
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()
        
        self.output = nn.Conv2d(256, num_anchor_points * 2, kernel_size=3, padding=1)
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input feature map [B, C, H, W]
            
        Returns:
            regression: Regression output [B, num_anchors * 2, H, W]
        """
        out = self.conv1(x)
        out = self.act1(out)
        
        out = self.conv2(out)
        out = self.act2(out)
        
        out = self.conv3(out)
        out = self.act3(out)
        
        out = self.conv4(out)
        out = self.act4(out)
        
        out = self.output(out)
        
        return out


class ClassificationModel(nn.Module):
    """Classification model for predicting classes"""
    
    def __init__(self, num_features_in, num_classes, num_anchors=1):
        super(ClassificationModel, self).__init__()
        
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        
        self.conv1 = nn.Conv2d(num_features_in, 256, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.output = nn.Conv2d(256, num_anchors * num_classes, kernel_size=3, padding=1)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """Forward pass
        
        Args:
            x: Input feature map [B, C, H, W]
            
        Returns:
            classification: Classification output [B, num_anchors * num_classes, H, W]
        """
        out = self.conv1(x)
        out = F.relu(out)
        
        out = self.conv2(out)
        out = F.relu(out)
        
        out = self.conv3(out)
        out = F.relu(out)
        
        out = self.conv4(out)
        out = F.relu(out)
        
        out = self.output(out)
        
        return out


class DetectionHead(nn.Module):
    """Detection head integrating regression and classification"""
    
    def __init__(self, input_dim, num_classes, num_anchor_points):
        super(DetectionHead, self).__init__()
        
        # Feature dimension reduction
        self.feature_conv = nn.Conv2d(input_dim, 1024, kernel_size=1)
        
        # Regression branch
        self.regression_model = RegressionModel(
            num_features_in=1024,
            num_anchor_points=num_anchor_points
        )
        
        # Classification branch
        self.classification_model = ClassificationModel(
            num_features_in=1024,
            num_classes=num_classes,
            num_anchor_points=num_anchor_points
        )
        
        # Anchor point generation
        self.anchor_points = AnchorPoints(
            pyramid_levels=[3],
            row=2,
            line=2
        )
        
    def forward(self, x, img_batch=None):
        """
        Forward pass
        
        Args:
            x: Input feature map [B, C, H, W]
            img_batch: Original image (for generating anchor points)
            
        Returns:
            dict: Dictionary containing regression and classification results
        """
        # Feature dimension reduction
        features = self.feature_conv(x)
        
        # Regression prediction
        regression = self.regression_model(features)
        
        # Classification prediction
        classification = self.classification_model(features)
        
        # Generate anchor points (if original image is provided)
        if img_batch is not None:
            anchor_points = self.anchor_points(img_batch)
            
            # Adjust regression output shape
            B, C, H, W = regression.shape
            regression = regression.view(B, -1, 2, H, W)
            regression = regression.permute(0, 1, 3, 4, 2).contiguous()
            regression = regression.view(B, -1, 2)
            
            # Adjust classification output shape
            B, C, H, W = classification.shape
            num_anchors = C // self.classification_model.num_classes
            classification = classification.view(B, num_anchors, self.classification_model.num_classes, H, W)
            classification = classification.permute(0, 1, 3, 4, 2).contiguous()
            classification = classification.view(B, -1, self.classification_model.num_classes)
            
            return {
                'regression': regression,
                'classification': classification,
                'anchor_points': anchor_points
            }
        
        return {
            'regression': regression,
            'classification': classification
        }