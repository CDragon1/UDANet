import torch
import numpy as np
import json
import os
from PIL import Image
import cv2
from typing import List, Dict, Tuple, Optional
from scipy.ndimage import gaussian_filter


def load_data_info(data_dir: str, annotation_file: str = None) -> List[Dict]:
    """
    加载数据信息
    
    Args:
        data_dir: 数据目录
        annotation_file: 标注文件路径（可选）
    
    Returns:
        数据信息列表，每个元素包含图像路径和标注信息
    """
    data_info = []
    
    if annotation_file and os.path.exists(annotation_file):
        # 从标注文件加载
        with open(annotation_file, 'r', encoding='utf-8') as f:
            annotations = json.load(f)
        
        for ann in annotations:
            img_path = os.path.join(data_dir, ann['image_name'])
            if os.path.exists(img_path):
                data_info.append({
                    'image_path': img_path,
                    'points': ann.get('points', []),
                    'count': ann.get('count', len(ann.get('points', []))),
                    'density_level': ann.get('density_level', None),
                    'image_id': ann.get('image_id', ann['image_name'])
                })
    else:
        # 从目录结构加载
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    img_path = os.path.join(root, file)
                    
                    # 尝试加载对应的标注文件
                    base_name = os.path.splitext(file)[0]
                    ann_files = [
                        os.path.join(root, f"{base_name}.json"),
                        os.path.join(root, f"{base_name}.txt"),
                        os.path.join(os.path.dirname(root), 'annotations', f"{base_name}.json")
                    ]
                    
                    points = []
                    count = 0
                    
                    for ann_path in ann_files:
                        if os.path.exists(ann_path):
                            if ann_path.endswith('.json'):
                                with open(ann_path, 'r', encoding='utf-8') as f:
                                    ann_data = json.load(f)
                                    points = ann_data.get('points', [])
                                    count = ann_data.get('count', len(points))
                            elif ann_path.endswith('.txt'):
                                points = load_points_from_txt(ann_path)
                                count = len(points)
                            break
                    
                    data_info.append({
                        'image_path': img_path,
                        'points': points,
                        'count': count,
                        'density_level': get_density_level(count),
                        'image_id': base_name
                    })
    
    return data_info


def load_points_from_txt(txt_file: str) -> List[List[float]]:
    """从TXT文件加载点坐标"""
    points = []
    with open(txt_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split(',')
                if len(parts) >= 2:
                    try:
                        x = float(parts[0])
                        y = float(parts[1])
                        points.append([x, y])
                    except ValueError:
                        continue
    return points


def get_density_level(count: int, thresholds: List[int] = None) -> int:
    """
    根据数量获取密度等级
    
    Args:
        count: 害虫数量
        thresholds: 密度阈值列表，默认为[10, 50, 100]
    
    Returns:
        密度等级 (0: 低密度, 1: 中密度, 2: 高密度, 3: 极高密度)
    """
    if thresholds is None:
        thresholds = [10, 50, 100]
    
    if count <= thresholds[0]:
        return 0  # 低密度
    elif count <= thresholds[1]:
        return 1  # 中密度
    elif count <= thresholds[2]:
        return 2  # 高密度
    else:
        return 3  # 极高密度


def generate_density_map(points: List[List[float]], 
                        image_shape: Tuple[int, int], 
                        sigma: float = 5.0,
                        method: str = 'gaussian') -> np.ndarray:
    """
    生成密度图
    
    Args:
        points: 点坐标列表
        image_shape: 图像形状 (height, width)
        sigma: 高斯滤波标准差
        method: 生成方法 ('gaussian', 'fixed', 'adaptive')
    
    Returns:
        密度图数组
    """
    height, width = image_shape
    density_map = np.zeros((height, width), dtype=np.float32)
    
    if len(points) == 0:
        return density_map
    
    # 在点位置添加高斯分布
    for point in points:
        x, y = int(point[0]), int(point[1])
        if 0 <= x < width and 0 <= y < height:
            if method == 'gaussian':
                # 固定高斯核
                kernel_size = int(3 * sigma)
                kernel = create_gaussian_kernel(kernel_size, sigma)
                
                # 计算放置位置
                x_min = max(0, x - kernel_size // 2)
                x_max = min(width, x + kernel_size // 2 + 1)
                y_min = max(0, y - kernel_size // 2)
                y_max = min(height, y + kernel_size // 2 + 1)
                
                # 裁剪核以适应边界
                kx_min = max(0, kernel_size // 2 - x)
                kx_max = min(kernel_size, kernel_size // 2 + (width - x))
                ky_min = max(0, kernel_size // 2 - y)
                ky_max = min(kernel_size, kernel_size // 2 + (height - y))
                
                if x_max > x_min and y_max > y_min and kx_max > kx_min and ky_max > ky_min:
                    density_map[y_min:y_max, x_min:x_max] += kernel[ky_min:ky_max, kx_min:kx_max]
            
            elif method == 'fixed':
                # 固定大小的核
                kernel_size = 15
                kernel = create_fixed_kernel(kernel_size)
                
                x_min = max(0, x - kernel_size // 2)
                x_max = min(width, x + kernel_size // 2 + 1)
                y_min = max(0, y - kernel_size // 2)
                y_max = min(height, y + kernel_size // 2 + 1)
                
                kx_min = max(0, kernel_size // 2 - x)
                kx_max = min(kernel_size, kernel_size // 2 + (width - x))
                ky_min = max(0, kernel_size // 2 - y)
                ky_max = min(kernel_size, kernel_size // 2 + (height - y))
                
                if x_max > x_min and y_max > y_min and kx_max > kx_min and ky_max > ky_min:
                    density_map[y_min:y_max, x_min:x_max] += kernel[ky_min:ky_max, kx_min:kx_max]
    
    return density_map


def create_gaussian_kernel(kernel_size: int, sigma: float) -> np.ndarray:
    """创建高斯核"""
    ax = np.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    return kernel / kernel.sum()


def create_fixed_kernel(kernel_size: int) -> np.ndarray:
    """创建固定核"""
    kernel = np.ones((kernel_size, kernel_size), dtype=np.float32)
    return kernel / kernel.sum()


def adaptive_density_map(points: List[List[float]], 
                        image_shape: Tuple[int, int],
                        k_neighbors: int = 5) -> np.ndarray:
    """
    自适应密度图生成
    
    Args:
        points: 点坐标列表
        image_shape: 图像形状
        k_neighbors: K近邻数量
    
    Returns:
        自适应密度图
    """
    height, width = image_shape
    density_map = np.zeros((height, width), dtype=np.float32)
    
    if len(points) == 0:
        return density_map
    
    # 计算每个点的自适应带宽
    points_array = np.array(points)
    for i, point in enumerate(points):
        x, y = int(point[0]), int(point[1])
        if 0 <= x < width and 0 <= y < height:
            # 计算到K个最近邻的距离
            distances = np.sqrt(np.sum((points_array - point)**2, axis=1))
            distances = np.sort(distances)[1:k_neighbors+1]  # 排除自身
            
            if len(distances) > 0:
                # 使用平均距离作为带宽
                sigma = np.mean(distances) / 2.0
                sigma = max(sigma, 2.0)  # 最小带宽
                
                # 生成高斯核
                kernel_size = int(3 * sigma)
                kernel = create_gaussian_kernel(kernel_size, sigma)
                
                # 添加到密度图
                x_min = max(0, x - kernel_size // 2)
                x_max = min(width, x + kernel_size // 2 + 1)
                y_min = max(0, y - kernel_size // 2)
                y_max = min(height, y + kernel_size // 2 + 1)
                
                kx_min = max(0, kernel_size // 2 - x)
                kx_max = min(kernel_size, kernel_size // 2 + (width - x))
                ky_min = max(0, kernel_size // 2 - y)
                ky_max = min(kernel_size, kernel_size // 2 + (height - y))
                
                if x_max > x_min and y_max > y_min and kx_max > kx_min and ky_max > ky_min:
                    density_map[y_min:y_max, x_min:x_max] += kernel[ky_min:ky_max, kx_min:kx_max]
    
    return density_map


def collate_fn(batch):
    """自定义批处理函数"""
    images = torch.stack([item['image'] for item in batch])
    
    # 处理点坐标（可变长度）
    points = [item['points'] for item in batch]
    
    # 处理密度图
    density_maps = torch.stack([item['density_map'] for item in batch])
    
    # 处理密度等级
    density_levels = torch.tensor([item['density_level'] for item in batch], dtype=torch.long)
    
    # 处理图像ID
    image_ids = [item['image_id'] for item in batch]
    
    return {
        'image': images,
        'points': points,
        'density_map': density_maps,
        'density_level': density_levels,
        'image_id': image_ids
    }


def save_annotation(image_path: str, points: List[List[float]], 
                   output_dir: str, format: str = 'json'):
    """保存标注文件"""
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    if format == 'json':
        ann_path = os.path.join(output_dir, f"{base_name}.json")
        annotation = {
            'image_name': os.path.basename(image_path),
            'points': points,
            'count': len(points),
            'density_level': get_density_level(len(points))
        }
        with open(ann_path, 'w', encoding='utf-8') as f:
            json.dump(annotation, f, indent=2, ensure_ascii=False)
    
    elif format == 'txt':
        ann_path = os.path.join(output_dir, f"{base_name}.txt")
        with open(ann_path, 'w', encoding='utf-8') as f:
            for point in points:
                f.write(f"{point[0]:.2f},{point[1]:.2f}\n")
    
    return ann_path


def split_dataset(data_info: List[Dict], train_ratio: float = 0.7, 
                 val_ratio: float = 0.15, test_ratio: float = 0.15,
                 random_seed: int = 42) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """划分数据集"""
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Train, validation, and test ratios must sum to 1.0"
    
    np.random.seed(random_seed)
    indices = np.random.permutation(len(data_info))
    
    n_train = int(len(data_info) * train_ratio)
    n_val = int(len(data_info) * val_ratio)
    
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]
    
    train_data = [data_info[i] for i in train_indices]
    val_data = [data_info[i] for i in val_indices]
    test_data = [data_info[i] for i in test_indices]
    
    return train_data, val_data, test_data


def calculate_dataset_statistics(data_info: List[Dict]) -> Dict:
    """计算数据集统计信息"""
    counts = [item['count'] for item in data_info]
    density_levels = [item['density_level'] for item in data_info]
    
    stats = {
        'total_samples': len(data_info),
        'total_objects': sum(counts),
        'avg_objects_per_image': np.mean(counts),
        'std_objects_per_image': np.std(counts),
        'min_objects': min(counts),
        'max_objects': max(counts),
        'median_objects': np.median(counts),
        'density_level_distribution': {
            0: density_levels.count(0),
            1: density_levels.count(1),
            2: density_levels.count(2),
            3: density_levels.count(3)
        }
    }
    
    return stats