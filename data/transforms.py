import torch
import torch.nn as nn
import torchvision.transforms as transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from PIL import Image
import cv2


class PestTransforms:
    """害虫检测和计数的数据变换类"""
    
    def __init__(self, config):
        self.config = config
        self.input_size = config.data.input_size
        self.mean = config.data.normalize.mean
        self.std = config.data.normalize.std
    
    def get_train_transforms(self):
        """获取训练数据变换"""
        return A.Compose([
            A.Resize(self.input_size, self.input_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandomRotate90(p=0.3),
            A.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1,
                p=0.5
            ),
            A.GaussianBlur(blur_limit=3, p=0.3),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
            A.Normalize(mean=self.mean, std=self.std),
            ToTensorV2()
        ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
    
    def get_val_transforms(self):
        """获取验证数据变换"""
        return A.Compose([
            A.Resize(self.input_size, self.input_size),
            A.Normalize(mean=self.mean, std=self.std),
            ToTensorV2()
        ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
    
    def get_test_transforms(self):
        """获取测试数据变换"""
        return A.Compose([
            A.Resize(self.input_size, self.input_size),
            A.Normalize(mean=self.mean, std=self.std),
            ToTensorV2()
        ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))


class AdvancedPestTransforms:
    """高级害虫检测数据变换"""
    
    def __init__(self, config):
        self.config = config
        self.input_size = config.data.input_size
        self.mean = config.data.normalize.mean
        self.std = config.data.normalize.std
    
    def get_train_transforms(self):
        """获取增强的训练数据变换"""
        return A.Compose([
            A.Resize(self.input_size, self.input_size),
            
            # 几何变换
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandomRotate90(p=0.3),
            A.Affine(
                scale=(0.8, 1.2),
                translate_percent=(-0.1, 0.1),
                rotate=(-15, 15),
                shear=(-5, 5),
                p=0.5
            ),
            A.Perspective(scale=(0.05, 0.1), p=0.3),
            
            # 颜色变换
            A.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.3,
                hue=0.1,
                p=0.6
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.3,
                contrast_limit=0.3,
                p=0.5
            ),
            A.HueSaturationValue(
                hue_shift_limit=20,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=0.4
            ),
            
            # 噪声和模糊
            A.GaussianBlur(blur_limit=(3, 7), p=0.4),
            A.MotionBlur(blur_limit=7, p=0.3),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.4),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.3),
            
            # 其他增强
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
            A.Equalize(p=0.2),
            A.RandomGamma(gamma_limit=(80, 120), p=0.3),
            
            A.Normalize(mean=self.mean, std=self.std),
            ToTensorV2()
        ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
    
    def get_val_transforms(self):
        """获取验证数据变换"""
        return A.Compose([
            A.Resize(self.input_size, self.input_size),
            A.Normalize(mean=self.mean, std=self.std),
            ToTensorV2()
        ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))


def get_train_transforms(config):
    """获取训练数据变换的工厂函数"""
    if config.data.get('use_advanced_augmentation', False):
        transforms = AdvancedPestTransforms(config)
    else:
        transforms = PestTransforms(config)
    
    return transforms.get_train_transforms()


def get_val_transforms(config):
    """获取验证数据变换的工厂函数"""
    transforms = PestTransforms(config)
    return transforms.get_val_transforms()


def get_test_transforms(config):
    """获取测试数据变换的工厂函数"""
    transforms = PestTransforms(config)
    return transforms.get_test_transforms()


class MixUp:
    """MixUp数据增强"""
    
    def __init__(self, alpha=0.2):
        self.alpha = alpha
    
    def __call__(self, batch, targets):
        """应用MixUp增强"""
        lam = np.random.beta(self.alpha, self.alpha)
        batch_size = batch.size(0)
        index = torch.randperm(batch_size)
        
        mixed_batch = lam * batch + (1 - lam) * batch[index, :]
        mixed_targets = []
        
        for i in range(batch_size):
            mixed_targets.append({
                'lam': lam,
                'target_a': targets[i],
                'target_b': targets[index[i]]
            })
        
        return mixed_batch, mixed_targets


class CutMix:
    """CutMix数据增强"""
    
    def __init__(self, alpha=1.0):
        self.alpha = alpha
    
    def __call__(self, batch, targets):
        """应用CutMix增强"""
        lam = np.random.beta(self.alpha, self.alpha)
        batch_size = batch.size(0)
        index = torch.randperm(batch_size)
        
        y1, x1, y2, x2 = self._rand_bbox(batch.size(), lam)
        mixed_batch = batch.clone()
        mixed_batch[:, :, y1:y2, x1:x2] = batch[index, :, y1:y2, x1:x2]
        
        # 调整lambda值
        lam = 1 - ((y2 - y1) * (x2 - x1) / (batch.size(-1) * batch.size(-2)))
        
        mixed_targets = []
        for i in range(batch_size):
            mixed_targets.append({
                'lam': lam,
                'target_a': targets[i],
                'target_b': targets[index[i]]
            })
        
        return mixed_batch, mixed_targets
    
    def _rand_bbox(self, size, lam):
        """生成随机边界框"""
        W = size[-1]
        H = size[-2]
        cut_rat = np.sqrt(1. - lam)
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)
        
        # 统一中心点
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        return bby1, bbx1, bby2, bbx2


def apply_mixup_or_cutmix(batch, targets, mixup_prob=0.5, cutmix_prob=0.5):
    """应用MixUp或CutMix增强"""
    if np.random.rand() < mixup_prob:
        mixup = MixUp(alpha=0.2)
        return mixup(batch, targets)
    elif np.random.rand() < cutmix_prob:
        cutmix = CutMix(alpha=1.0)
        return cutmix(batch, targets)
    else:
        return batch, targets