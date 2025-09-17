import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import cv2
from scipy.ndimage import gaussian_filter


class PestDataset(Dataset):
    """害虫检测与计数数据集"""
    
    def __init__(self, data_root, classes, transform=None, target_size=(512, 512), sigma=5.0, density_threshold=10):
        """
        初始化数据集
        
        Args:
            data_root: 数据根目录
            classes: 害虫类别列表
            transform: 数据变换
            target_size: 目标图像尺寸
            sigma: 高斯核标准差
            density_threshold: 密度阈值
        """
        self.data_root = data_root
        self.classes = classes
        self.transform = transform
        self.target_size = target_size
        self.sigma = sigma
        self.density_threshold = density_threshold
        
        # 收集所有样本
        self.samples = []
        for class_name in classes:
            class_path = os.path.join(data_root, class_name)
            if os.path.exists(class_path):
                image_dir = os.path.join(class_path, 'images')
                label_dir = os.path.join(class_path, 'labels')
                
                if os.path.exists(image_dir) and os.path.exists(label_dir):
                    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
                    
                    for img_file in image_files:
                        img_path = os.path.join(image_dir, img_file)
                        label_path = os.path.join(label_dir, img_file.replace('.jpg', '.txt').replace('.png', '.txt'))
                        
                        if os.path.exists(label_path):
                            self.samples.append({
                                'image_path': img_path,
                                'label_path': label_path,
                                'class_name': class_name
                            })
        
        print(f"Loaded {len(self.samples)} samples from {len(classes)} classes")
        
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        """获取单个样本"""
        sample = self.samples[idx]
        
        # 加载图像
        image = Image.open(sample['image_path']).convert('RGB')
        original_size = image.size  # (W, H)
        
        # 加载标注
        points = self.load_labels(sample['label_path'], original_size)
        
        # 调整图像尺寸
        image = image.resize(self.target_size, Image.Resampling.LANCZOS)
        
        # 调整点坐标
        scale_x = self.target_size[0] / original_size[0]
        scale_y = self.target_size[1] / original_size[1]
        
        adjusted_points = []
        for point in points:
            x, y, class_id = point
            new_x = x * scale_x
            new_y = y * scale_y
            adjusted_points.append([new_x, new_y, class_id])
        
        # 生成密度图
        density_map = self.generate_density_map(adjusted_points, self.target_size[::-1])
        
        # 计算密度等级
        density_level = 0 if len(adjusted_points) <= self.density_threshold else 1
        
        # 转换为numpy数组
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1)  # HWC to CHW
        
        density_map = torch.from_numpy(density_map).unsqueeze(0)  # Add channel dimension
        
        # 准备点数据
        points_tensor = torch.tensor(adjusted_points, dtype=torch.float32)
        
        # 应用变换
        if self.transform:
            image = self.transform(image)
        
        return {
            'image': image,
            'points': points_tensor,
            'density_map': density_map,
            'density_level': torch.tensor(density_level, dtype=torch.long),
            'num_points': torch.tensor(len(adjusted_points), dtype=torch.float32),
            'image_path': sample['image_path'],
            'class_name': sample['class_name']
        }
    
    def load_labels(self, label_path, image_size):
        """加载标注文件"""
        points = []
        
        try:
            with open(label_path, 'r') as f:
                lines = f.readlines()
                
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 3:
                    x = float(parts[0])
                    y = float(parts[1])
                    class_id = int(parts[2])
                    points.append([x, y, class_id])
                    
        except Exception as e:
            print(f"Error loading label file {label_path}: {e}")
            
        return points
    
    def generate_density_map(self, points, image_shape, sigma=None):
        """生成密度图"""
        if sigma is None:
            sigma = self.sigma
            
        H, W = image_shape
        density_map = np.zeros((H, W), dtype=np.float32)
        
        for point in points:
            x, y, _ = point
            x_int = int(x)
            y_int = int(y)
            
            if 0 <= x_int < W and 0 <= y_int < H:
                density_map[y_int, x_int] += 1
        
        # 应用高斯滤波
        if len(points) > 0:
            density_map = gaussian_filter(density_map, sigma=sigma)
            
        return density_map


class PestDataModule:
    """数据模块，用于管理训练和验证数据"""
    
    def __init__(self, config):
        self.config = config
        self.data_root = config.data.root_path
        self.classes = config.data.classes
        self.batch_size = config.training.batch_size
        self.num_workers = config.training.num_workers
        self.image_size = config.data.image_size
        
    def setup(self, stage=None):
        """设置数据集"""
        # 加载所有样本信息
        all_samples = []
        for class_name in self.classes:
            class_path = os.path.join(self.data_root, class_name)
            if os.path.exists(class_path):
                image_dir = os.path.join(class_path, 'images')
                label_dir = os.path.join(class_path, 'labels')
                
                if os.path.exists(image_dir) and os.path.exists(label_dir):
                    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
                    
                    for img_file in image_files:
                        img_path = os.path.join(image_dir, img_file)
                        label_path = os.path.join(label_dir, img_file.replace('.jpg', '.txt').replace('.png', '.txt'))
                        
                        if os.path.exists(label_path):
                            all_samples.append({
                                'image_path': img_path,
                                'label_path': label_path,
                                'class_name': class_name
                            })
        
        # 划分数据集
        total_samples = len(all_samples)
        train_size = int(total_samples * config.data.train_split)
        val_size = int(total_samples * config.data.val_split)
        test_size = total_samples - train_size - val_size
        
        indices = torch.randperm(total_samples)
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]
        
        # 创建数据集
        self.train_dataset = PestDataset(
            data_root=self.data_root,
            classes=self.classes,
            transform=get_train_transforms(self.config),
            target_size=tuple(self.image_size)
        )
        self.train_dataset.samples = [all_samples[i] for i in train_indices]
        
        self.val_dataset = PestDataset(
            data_root=self.data_root,
            classes=self.classes,
            transform=get_val_transforms(self.config),
            target_size=tuple(self.image_size)
        )
        self.val_dataset.samples = [all_samples[i] for i in val_indices]
        
        self.test_dataset = PestDataset(
            data_root=self.data_root,
            classes=self.classes,
            transform=get_val_transforms(self.config),
            target_size=tuple(self.image_size)
        )
        self.test_dataset.samples = [all_samples[i] for i in test_indices]
        
        print(f"Dataset split: Train={len(self.train_dataset)}, Val={len(self.val_dataset)}, Test={len(self.test_dataset)}")
        
    def train_dataloader(self):
        """训练数据加载器"""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=collate_fn
        )
        
    def val_dataloader(self):
        """验证数据加载器"""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=collate_fn
        )
        
    def test_dataloader(self):
        """测试数据加载器"""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=collate_fn
        )


def collate_fn(batch):
    """自定义批处理函数"""
    # 分离不同类型的数据
    images = torch.stack([item['image'] for item in batch])
    density_maps = torch.stack([item['density_map'] for item in batch])
    density_levels = torch.stack([item['density_level'] for item in batch])
    num_points = torch.stack([item['num_points'] for item in batch])
    
    # 点数据需要特殊处理（因为数量可能不同）
    points = [item['points'] for item in batch]
    
    # 其他信息
    image_paths = [item['image_path'] for item in batch]
    class_names = [item['class_name'] for item in batch]
    
    return {
        'image': images,
        'points': points,
        'density_map': density_maps,
        'density_level': density_levels,
        'num_points': num_points,
        'image_path': image_paths,
        'class_name': class_names
    }


def get_train_transforms(config):
    """获取训练变换"""
    from torchvision import transforms
    
    transform_list = []
    
    # 基础变换
    transform_list.extend([
        transforms.ToPILImage(),
        transforms.ColorJitter(
            brightness=config.augmentation.brightness,
            contrast=config.augmentation.contrast,
            saturation=config.augmentation.saturation,
            hue=config.augmentation.hue
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return transforms.Compose(transform_list)


def get_val_transforms(config):
    """获取验证变换"""
    from torchvision import transforms
    
    transform_list = [
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
    
    return transforms.Compose(transform_list)