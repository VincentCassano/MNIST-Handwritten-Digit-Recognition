# 模型定义文件
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    """
    简单的CNN模型，适合MNIST基础识别
    结构：2个卷积层 + 2个全连接层
    """
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        # 第一个卷积层：输入1通道(灰度图)，输出32通道，卷积核3x3，步长1，填充1
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        # 第二个卷积层：输入32通道，输出64通道
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        # 池化层：2x2最大池化
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # 全连接层1：输入64*7*7=3136，输出128
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        # 全连接层2：输入128，输出num_classes(10)
        self.fc2 = nn.Linear(128, num_classes)
        # Dropout层防止过拟合
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x (torch.Tensor): 输入张量，形状为 [batch_size, 1, 28, 28]
        
        Returns:
            torch.Tensor: 输出张量，形状为 [batch_size, num_classes]
        """
        # 输入形状: [batch_size, 1, 28, 28]
        x = self.pool(F.relu(self.conv1(x)))
        # 输出形状: [batch_size, 32, 14, 14]
        x = self.pool(F.relu(self.conv2(x)))
        # 输出形状: [batch_size, 64, 7, 7]
        x = x.view(-1, 64 * 7 * 7)  # 展平
        # 输出形状: [batch_size, 3136]
        x = self.dropout(F.relu(self.fc1(x)))
        # 输出形状: [batch_size, 128]
        x = self.fc2(x)
        # 输出形状: [batch_size, 10]
        return x


class MediumCNN(nn.Module):
    """
    中等复杂度的CNN模型，适合更高精度的MNIST识别
    结构：3个卷积层 + 2个全连接层
    """
    def __init__(self, num_classes=10):
        super(MediumCNN, self).__init__()
        # 卷积层序列
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),  # 批量归一化
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1)  # 添加padding以保持维度
        )
        
        # 全连接层
        self.fc_layers = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x (torch.Tensor): 输入张量，形状为 [batch_size, 1, 28, 28]
        
        Returns:
            torch.Tensor: 输出张量，形状为 [batch_size, num_classes]
        """
        x = self.conv_layers(x)
        x = x.view(-1, 128 * 4 * 4)
        x = self.fc_layers(x)
        return x


class AdvancedCNN(nn.Module):
    """
    高级CNN模型，包含残差连接和更复杂的网络结构
    适合需要最高精度的MNIST识别任务
    """
    def __init__(self, num_classes=10):
        super(AdvancedCNN, self).__init__()
        
        # 初始卷积层
        self.initial_conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        # 残差块1
        self.residual_block1 = self._make_residual_block(64, 64)
        # 残差块2（通道数加倍）
        self.residual_block2 = self._make_residual_block(64, 128, downsample=True)
        # 残差块3
        self.residual_block3 = self._make_residual_block(128, 128)
        # 残差块4（通道数加倍）
        self.residual_block4 = self._make_residual_block(128, 256, downsample=True)
        # 全局平均池化
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    
    def _make_residual_block(self, in_channels, out_channels, downsample=False):
        """创建残差块"""
        stride = 2 if downsample else 1
        
        # 主路径
        main_path = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        
        # 快捷路径（如果需要降采样或通道数改变）
        shortcut = nn.Sequential()
        if downsample or in_channels != out_channels:
            shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        
        return ResidualBlock(main_path, shortcut)
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x (torch.Tensor): 输入张量，形状为 [batch_size, 1, 28, 28]
        
        Returns:
            torch.Tensor: 输出张量，形状为 [batch_size, num_classes]
        """
        x = self.initial_conv(x)
        x = self.residual_block1(x)
        x = self.residual_block2(x)
        x = self.residual_block3(x)
        x = self.residual_block4(x)
        x = self.global_pool(x)
        x = x.view(-1, 256)
        x = self.classifier(x)
        return x


class ResidualBlock(nn.Module):
    """残差块实现"""
    def __init__(self, main_path, shortcut):
        super(ResidualBlock, self).__init__()
        self.main_path = main_path
        self.shortcut = shortcut
    
    def forward(self, x):
        residual = x
        x = self.main_path(x)
        x += self.shortcut(residual)
        x = F.relu(x)
        return x


def get_model(model_name, num_classes=10):
    """
    根据模型名称获取相应的模型实例
    
    Args:
        model_name (str): 模型名称，可选 'simple', 'medium', 'advanced'
        num_classes (int): 分类数量，默认为10
    
    Returns:
        nn.Module: 初始化的模型实例
    """
    model_dict = {
        'simple': SimpleCNN,
        'medium': MediumCNN,
        'advanced': AdvancedCNN
    }
    
    if model_name not in model_dict:
        raise ValueError(f"Unknown model name: {model_name}. Available options: {list(model_dict.keys())}")
    
    return model_dict[model_name](num_classes)


def count_parameters(model):
    """
    计算模型的参数数量
    
    Args:
        model (nn.Module): 模型实例
    
    Returns:
        int: 参数数量
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)