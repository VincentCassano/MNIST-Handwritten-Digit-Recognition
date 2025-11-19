# 数据加载和预处理模块

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, SubsetRandomSampler
import numpy as np

# 尝试导入matplotlib，如果失败则设置为None
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
    print("✅ 成功导入matplotlib")
except ImportError:
    plt = None
    HAS_MATPLOTLIB = False
    print("⚠️ 无法导入matplotlib，将跳过可视化功能")

from typing import Dict, Tuple, Optional, List


class MNISTDataLoader:
    """
    MNIST数据集加载器类，处理数据的加载、预处理和增强
    """
    
    def __init__(self, config: Dict):
        """
        初始化数据加载器
        
        Args:
            config (Dict): 数据加载配置参数
        """
        self.config = config
        self.batch_size = config['batch_size']
        self.num_workers = config.get('num_workers', 4)
        self.pin_memory = config.get('pin_memory', True)
        self.use_cuda = torch.cuda.is_available()
        
        # 创建数据变换
        self.train_transform = self._create_train_transforms()
        self.test_transform = self._create_test_transforms()
        
        # 加载数据集
        self.train_dataset, self.test_dataset = self._load_datasets()
        
    def _create_train_transforms(self) -> transforms.Compose:
        """
        创建训练集的数据变换（包含数据增强）
        """
        transform_list = [
            # 随机旋转
            transforms.RandomRotation(self.config.get('rotation_degree', 10)),
            # 随机平移和缩放
            transforms.RandomAffine(
                degrees=0,
                translate=self.config.get('translation', (0.1, 0.1)),
                scale=self.config.get('scale_range', (0.9, 1.1))
            ),
            # 随机水平翻转（对于MNIST影响不大，但保持一致性）
            transforms.RandomHorizontalFlip(),
            # 随机裁剪
            transforms.RandomCrop(
                size=28,
                padding=self.config.get('crop_padding', 2),
                padding_mode=self.config.get('padding_mode', 'edge')
            ),
            # 转换为张量
            transforms.ToTensor(),
            # 标准化
            transforms.Normalize(
                mean=self.config.get('mean', (0.1307,)),
                std=self.config.get('std', (0.3081,))
            )
        ]
        
        # 如果配置了添加噪声
        if self.config.get('add_noise', False):
            # 定义可序列化的噪声添加函数
            def add_noise(x):
                return x + torch.randn_like(x) * self.config.get('noise_std', 0.05)
            
            transform_list.append(transforms.Lambda(add_noise))
        
        return transforms.Compose(transform_list)
    
    def _create_test_transforms(self) -> transforms.Compose:
        """
        创建测试集的数据变换（不包含数据增强，只进行标准化）
        """
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=self.config.get('mean', (0.1307,)),
                std=self.config.get('std', (0.3081,))
            )
        ])
    
    def _load_datasets(self) -> Tuple[torchvision.datasets.MNIST, torchvision.datasets.MNIST]:
        """
        加载MNIST训练集和测试集
        """
        # 数据集下载目录
        data_dir = self.config.get('data_dir', './data')
        
        # 加载训练集
        train_dataset = torchvision.datasets.MNIST(
            root=data_dir,
            train=True,
            download=True,
            transform=self.train_transform
        )
        
        # 加载测试集
        test_dataset = torchvision.datasets.MNIST(
            root=data_dir,
            train=False,
            download=True,
            transform=self.test_transform
        )
        
        return train_dataset, test_dataset
    
    def get_train_val_loaders(self, val_ratio: float = 0.1) -> Tuple[DataLoader, DataLoader]:
        """
        获取训练集和验证集的数据加载器
        
        Args:
            val_ratio (float): 验证集比例
        
        Returns:
            Tuple[DataLoader, DataLoader]: 训练集加载器和验证集加载器
        """
        # 获取数据集大小
        train_size = len(self.train_dataset)
        val_size = int(train_size * val_ratio)
        
        # 创建索引
        indices = list(range(train_size))
        np.random.shuffle(indices)
        
        # 分割训练集和验证集索引
        train_indices, val_indices = indices[val_size:], indices[:val_size]
        
        # 创建采样器
        train_sampler = SubsetRandomSampler(train_indices)
        val_sampler = SubsetRandomSampler(val_indices)
        
        # 创建数据加载器
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=train_sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
        
        val_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=val_sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
        
        return train_loader, val_loader
    
    def get_test_loader(self) -> DataLoader:
        """
        获取测试集的数据加载器
        
        Returns:
            DataLoader: 测试集加载器
        """
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
    
    def get_train_loader(self) -> DataLoader:
        """
        获取整个训练集的数据加载器（不分割验证集）
        
        Returns:
            DataLoader: 训练集加载器
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
    
    def get_dynamic_batch_size(self, model_size_mb: float, gpu_mem_mb: Optional[float] = None) -> int:
        """
        根据模型大小和GPU内存动态调整批量大小
        
        Args:
            model_size_mb (float): 模型大小（MB）
            gpu_mem_mb (Optional[float]): GPU内存大小（MB），如果为None则自动检测
        
        Returns:
            int: 推荐的批量大小
        """
        if not self.use_cuda or gpu_mem_mb is None:
            gpu_mem_mb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
        
        # 保守估计：每个样本大约需要的内存是模型大小的2倍（前向+反向）
        # 留出20%的GPU内存给系统和其他操作
        available_mem_mb = gpu_mem_mb * 0.8
        estimated_sample_mem_mb = model_size_mb * 2
        
        # 计算最大可能的批量大小
        max_batch_size = int(available_mem_mb / estimated_sample_mem_mb)
        
        # 确保批量大小在合理范围内
        min_batch_size = 8
        max_batch_size = max(min_batch_size, min(max_batch_size, 1024))
        
        # 返回2的幂次的批量大小，通常会有更好的性能
        power_of_two = 2 ** int(np.log2(max_batch_size))
        return min(power_of_two, max_batch_size)


def visualize_samples(data_loader: DataLoader, num_samples: int = 25, figsize: Tuple[int, int] = (10, 10)):
    """
    可视化数据集中的样本
    
    Args:
        data_loader (DataLoader): 数据加载器
        num_samples (int): 要可视化的样本数量
        figsize (Tuple[int, int]): 图表大小
    """
    if not HAS_MATPLOTLIB:
        print("⚠️ matplotlib不可用，跳过可视化")
        return None
    
    # 获取一个批次的数据
    images, labels = next(iter(data_loader))
    
    # 限制样本数量
    num_samples = min(num_samples, len(images))
    
    # 创建子图
    fig, axes = plt.subplots(int(np.ceil(num_samples / 5)), 5, figsize=figsize)
    axes = axes.flatten()
    
    # 显示样本
    for i in range(num_samples):
        # 反标准化图像
        image = images[i].numpy().squeeze()
        mean = data_loader.dataset.transform.transforms[-1].mean[0]
        std = data_loader.dataset.transform.transforms[-1].std[0]
        image = image * std + mean
        
        axes[i].imshow(image, cmap='gray')
        axes[i].set_title(f"Label: {labels[i].item()}")
        axes[i].axis('off')
    
    # 隐藏多余的子图
    for i in range(num_samples, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    return fig


def analyze_class_distribution(dataset: torchvision.datasets.MNIST) -> Dict[int, int]:
    """
    分析数据集中各个类别的分布
    
    Args:
        dataset (torchvision.datasets.MNIST): MNIST数据集
    
    Returns:
        Dict[int, int]: 类别分布字典
    """
    # 获取所有标签
    labels = []
    for _, label in dataset:
        labels.append(label)
    
    # 计算每个类别的数量
    distribution = {}
    for label in set(labels):
        distribution[label] = labels.count(label)
    
    return distribution


def plot_class_distribution(distribution: Dict[int, int], title: str = "Class Distribution") -> Optional[object]:
    """
    绘制类别分布图
    
    Args:
        distribution: 类别分布字典 {类别: 数量}
        title: 图表标题
    
    Returns:
        Optional[object]: 生成的图表，如果matplotlib不可用则返回None
    """
    """
    绘制类别分布图
    
    Args:
        distribution (Dict[int, int]): 类别分布字典
        title (str): 图表标题
    
    Returns:
        Optional[plt.Figure]: 生成的图表，如果matplotlib不可用则返回None
    """
    if not HAS_MATPLOTLIB:
        print("⚠️ matplotlib不可用，跳过可视化")
        return None
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 排序类别
    sorted_classes = sorted(distribution.keys())
    counts = [distribution[cls] for cls in sorted_classes]
    
    # 绘制柱状图
    bars = ax.bar(sorted_classes, counts, color='skyblue')
    
    # 添加标签和标题
    ax.set_xlabel('Class')
    ax.set_ylabel('Count')
    ax.set_title(title)
    ax.set_xticks(sorted_classes)
    
    # 在柱状图上添加数值标签
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 100,
                f'{int(height)}', ha='center', va='bottom')
    
    plt.tight_layout()
    return fig


# 示例配置
DEFAULT_DATA_CONFIG = {
    'batch_size': 64,
    'num_workers': 4,
    'pin_memory': True,
    'rotation_degree': 10,
    'translation': (0.1, 0.1),
    'scale_range': (0.9, 1.1),
    'crop_padding': 2,
    'padding_mode': 'edge',
    'mean': (0.1307,),
    'std': (0.3081,),
    'add_noise': False,
    'noise_std': 0.05,
    'data_dir': './data'
}


if __name__ == "__main__":
    # 测试数据加载器
    data_loader = MNISTDataLoader(DEFAULT_DATA_CONFIG)
    
    # 获取训练集和验证集加载器
    train_loader, val_loader = data_loader.get_train_val_loaders()
    test_loader = data_loader.get_test_loader()
    
    print(f"训练集批次数量: {len(train_loader)}")
    print(f"验证集批次数量: {len(val_loader)}")
    print(f"测试集批次数量: {len(test_loader)}")
    
    # 分析并绘制类别分布
    train_dist = analyze_class_distribution(data_loader.train_dataset)
    test_dist = analyze_class_distribution(data_loader.test_dataset)
    
    print("\n训练集类别分布:")
    for cls, count in sorted(train_dist.items()):
        print(f"类别 {cls}: {count} 样本")
    
    print("\n测试集类别分布:")
    for cls, count in sorted(test_dist.items()):
        print(f"类别 {cls}: {count} 样本")
    
    # 可视化样本
    print("\n可视化样本（需要matplotlib）...")
    # 注意：在非交互环境中运行时，可能需要保存图片而不是显示
    # fig = visualize_samples(train_loader)
    # fig.savefig('train_samples.png')
    # 
    # fig = plot_class_distribution(train_dist, "MNIST 训练集类别分布")
    # fig.savefig('train_distribution.png')