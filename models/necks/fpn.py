import torch
import torch.nn as nn
import torch.nn.functional as F


class FPN(nn.Module):
    """
    Feature Pyramid Network (FPN)
    用于多尺度特征融合
    """
    
    def __init__(self, in_channels, out_channels, num_levels=3):
        super(FPN, self).__init__()
        
        self.num_levels = num_levels
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # 横向连接 (lateral connections)
        self.lateral_convs = nn.ModuleList()
        for i in range(num_levels):
            self.lateral_convs.append(
                nn.Conv2d(in_channels[i], out_channels, kernel_size=1)
            )
        
        # 输出卷积层
        self.fpn_convs = nn.ModuleList()
        for i in range(num_levels):
            self.fpn_convs.append(
                nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
            )
            
        # 初始化权重
        self._init_weights()
        
    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
    def forward(self, inputs):
        """
        前向传播
        
        Args:
            inputs: 多尺度特征列表 [P2, P3, P4, ...]
            
        Returns:
            outputs: FPN输出特征列表
        """
        assert len(inputs) == self.num_levels
        
        # 构建横向连接特征
        laterals = []
        for i, lateral_conv in enumerate(self.lateral_convs):
            laterals.append(lateral_conv(inputs[i]))
        
        # 自顶向下融合
        for i in range(self.num_levels - 1, 0, -1):
            # 上采样低层特征
            upsampled = F.interpolate(
                laterals[i], 
                size=laterals[i-1].shape[2:], 
                mode='nearest'
            )
            # 逐元素相加
            laterals[i-1] = laterals[i-1] + upsampled
        
        # 输出卷积
        outputs = []
        for i, fpn_conv in enumerate(self.fpn_convs):
            outputs.append(fpn_conv(laterals[i]))
            
        return outputs


class FPNWithBottomUp(nn.Module):
    """
    带有自底向上路径增强的FPN
    """
    
    def __init__(self, in_channels, out_channels, num_levels=3):
        super(FPNWithBottomUp, self).__init__()
        
        self.num_levels = num_levels
        
        # 标准FPN
        self.fpn = FPN(in_channels, out_channels, num_levels)
        
        # 自底向上路径
        self.downsample_convs = nn.ModuleList()
        for i in range(num_levels - 1):
            self.downsample_convs.append(
                nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
            )
            
        # 最终融合层
        self.fusion_convs = nn.ModuleList()
        for i in range(num_levels):
            self.fusion_convs.append(
                nn.Sequential(
                    nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
            )
            
    def forward(self, inputs):
        """
        前向传播
        
        Args:
            inputs: 多尺度特征列表
            
        Returns:
            outputs: 增强后的特征列表
        """
        # 标准FPN处理
        fpn_outputs = self.fpn(inputs)
        
        # 自底向上增强
        bottom_up_outputs = []
        
        # 第一层保持不变
        bottom_up_outputs.append(fpn_outputs[0])
        
        # 逐层处理
        for i in range(1, self.num_levels):
            # 下采样低层特征
            downsampled = self.downsample_convs[i-1](bottom_up_outputs[i-1])
            
            # 融合特征
            fused = torch.cat([fpn_outputs[i], downsampled], dim=1)
            fused = self.fusion_convs[i](fused)
            
            bottom_up_outputs.append(fused)
            
        return bottom_up_outputs


class PANet(nn.Module):
    """
    Path Aggregation Network (PANet)
    另一种特征融合架构
    """
    
    def __init__(self, in_channels, out_channels, num_levels=3):
        super(PANet, self).__init__()
        
        self.num_levels = num_levels
        
        # 首先使用FPN进行自顶向下融合
        self.fpn = FPN(in_channels, out_channels, num_levels)
        
        # 然后使用PANet进行自底向上融合
        self.pafpn_convs = nn.ModuleList()
        for i in range(num_levels - 1):
            self.pafpn_convs.append(
                nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
            )
            
    def forward(self, inputs):
        """
        前向传播
        
        Args:
            inputs: 多尺度特征列表
            
        Returns:
            outputs: PANet输出特征列表
        """
        # FPN处理
        inter_outs = self.fpn(inputs)
        
        # PANet自底向上处理
        for i in range(self.num_levels - 1):
            # 下采样低层特征并融合
            inter_outs[i + 1] = inter_outs[i + 1] + self.pafpn_convs[i](inter_outs[i])
            
        return inter_outs