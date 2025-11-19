# 可视化工具

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import random
from PIL import Image
from torchvision import transforms

# 设置matplotlib字体支持中文
def setup_chinese_font():
    """设置matplotlib支持中文显示"""
    try:
        # 尝试多个可能的中文字体，按优先级排序
        chinese_fonts = ['SimHei', 'Microsoft YaHei', 'Heiti TC', 'Arial Unicode MS', 'SimSun']
        
        # 尝试每个字体
        for font in chinese_fonts:
            try:
                plt.rcParams['font.sans-serif'] = [font]  # 用来正常显示中文标签
                plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
                # 测试字体是否可用
                plt.figure(figsize=(1, 1))
                plt.text(0.5, 0.5, '测试中文', ha='center', va='center')
                plt.close()
                print(f"✅ 成功设置中文字体: {font}")
                return True
            except:
                continue
        
        # 如果都失败，尝试使用系统默认字体
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
        plt.rcParams['axes.unicode_minus'] = False
        print("⚠️ 未找到合适的中文字体，将使用默认字体，中文可能无法正确显示")
    except Exception as e:
        print(f"⚠️ 设置字体时出错: {e}")

# 初始化中文字体支持
setup_chinese_font()

# 注释掉streamlit导入，使用matplotlib实现可视化功能
# import streamlit as st

from model_def import get_model
from data_loader import MNISTDataLoader

# 设置随机种子
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

# 可视化训练过程中的准确率和损失曲线
def plot_training_history(history, save_path=None, show=True):
    """可视化训练历史记录"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # 绘制准确率曲线
    axes[0].plot(history['train_acc'], label='训练准确率')
    axes[0].plot(history['val_acc'], label='验证准确率')
    axes[0].set_title('训练和验证准确率')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('准确率')
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.7)
    
    # 绘制损失曲线
    axes[1].plot(history['train_loss'], label='训练损失')
    axes[1].plot(history['val_loss'], label='验证损失')
    axes[1].set_title('训练和验证损失')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('损失')
    axes[1].legend()
    axes[1].grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'训练历史可视化已保存至: {save_path}')
    
    if show:
        plt.show()
    else:
        plt.close()

# 从文件加载训练历史
def load_training_history(history_path):
    """从.npy文件加载训练历史"""
    try:
        history = np.load(history_path, allow_pickle=True).item()
        return history
    except Exception as e:
        print(f'加载训练历史失败: {e}')
        return None

# 可视化模型预测结果
def visualize_predictions(model, test_loader, device, num_samples=10, save_path=None):
    """可视化模型对测试集样本的预测结果"""
    model.eval()
    
    # 获取一个批次的测试数据
    data_iter = iter(test_loader)
    images, labels = next(data_iter)
    
    # 选择有限数量的样本
    num_to_show = min(num_samples, len(images))
    images = images[:num_to_show].to(device)
    labels = labels[:num_to_show].cpu().numpy()
    
    # 进行预测
    with torch.no_grad():
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        confidence = torch.max(probs, 1)[0].cpu().numpy()
    
    preds = preds.cpu().numpy()
    images = images.cpu()
    
    # 创建子图
    rows = (num_to_show + 2) // 3  # 每行显示3个样本
    cols = min(3, num_to_show)
    plt.figure(figsize=(cols * 4, rows * 4))
    
    for i in range(num_to_show):
        plt.subplot(rows, cols, i + 1)
        img = images[i].squeeze().numpy()
        plt.imshow(img, cmap='gray')
        
        # 根据预测是否正确设置标题颜色
        color = 'green' if preds[i] == labels[i] else 'red'
        title = f'True: {labels[i]}\nPred: {preds[i]}\nConf: {confidence[i]:.2%}'
        plt.title(title, fontsize=10, color=color)
        plt.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'预测结果可视化已保存至: {save_path}')
    
    plt.show()

# 可视化卷积层特征图
def visualize_feature_maps(model, input_image, layer_indices=[0], save_path=None):
    """可视化模型中指定卷积层的特征图"""
    # 定义钩子函数来捕获特征图
    feature_maps = []
    
    def get_features(name):
        def hook(model, input, output):
            feature_maps.append(output.detach())
        return hook
    
    # 注册钩子
    hooks = []
    conv_layers = []
    
    # 查找所有卷积层
    def find_conv_layers(module, prefix=''):
        for name, layer in module.named_children():
            if isinstance(layer, torch.nn.Conv2d):
                conv_layers.append((f'{prefix}.{name}' if prefix else name, layer))
            find_conv_layers(layer, f'{prefix}.{name}' if prefix else name)
    
    find_conv_layers(model)
    
    if not conv_layers:
        print("未找到卷积层")
        return
    
    # 为指定的卷积层注册钩子
    for idx in layer_indices:
        if 0 <= idx < len(conv_layers):
            layer_name, layer = conv_layers[idx]
            hook = layer.register_forward_hook(get_features(layer_name))
            hooks.append(hook)
    
    # 前向传播
    model.eval()
    with torch.no_grad():
        _ = model(input_image.unsqueeze(0))
    
    # 移除钩子
    for hook in hooks:
        hook.remove()
    
    # 可视化特征图
    for i, features in enumerate(feature_maps):
        num_features = features.size(1)  # 特征图数量
        num_to_show = min(16, num_features)  # 只显示前16个特征图
        
        rows = int(np.ceil(np.sqrt(num_to_show)))
        cols = int(np.ceil(num_to_show / rows))
        
        plt.figure(figsize=(cols * 2, rows * 2))
        plt.suptitle(f'Layer {layer_indices[i]}: {num_features} feature maps', fontsize=12)
        
        for j in range(num_to_show):
            plt.subplot(rows, cols, j + 1)
            plt.imshow(features[0, j].cpu().numpy(), cmap='viridis')
            plt.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            save_file = f'{save_path}_layer_{layer_indices[i]}.png'
            plt.savefig(save_file, dpi=300, bbox_inches='tight')
            print(f'特征图已保存至: {save_file}')
        
        plt.show()

# 使用matplotlib创建简单的演示应用
def run_matplotlib_demo(model_path='models/best_model.pth', model_type='medium'):
    """使用matplotlib创建简单的演示应用（替代streamlit）"""
    print("运行MNIST手写数字识别演示（matplotlib版）")
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载模型
    try:
        model = get_model(model_type).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print("模型加载成功")
    except Exception as e:
        print(f"模型加载失败: {e}")
        return
    
    # 数据转换
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # 加载测试数据并预测几个样本（使用config字典）
    data_config = {
        'batch_size': 5,
        'num_workers': 0,
        'pin_memory': False,
        'mean': (0.1307,),
        'std': (0.3081,),
        'data_dir': 'data'
    }
    data_loader = MNISTDataLoader(data_config)
    test_loader = data_loader.get_test_loader()
    
    # 获取一个批次的数据
    data_iter = iter(test_loader)
    images, labels = next(data_iter)
    
    # 预测并可视化
    with torch.no_grad():
        outputs = model(images[:5].to(device))
        probabilities = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()
        predicted_classes = np.argmax(probabilities, axis=1)
    
    # 创建可视化结果
    fig, axes = plt.subplots(5, 2, figsize=(12, 20))
    
    for i in range(5):
        # 显示数字图像
        axes[i, 0].imshow(images[i].squeeze().numpy(), cmap='gray')
        axes[i, 0].set_title(f'样本 {i+1}\n真实标签: {labels[i].item()}\n预测标签: {predicted_classes[i]}')
        axes[i, 0].axis('off')
        
        # 显示概率分布
        axes[i, 1].bar(range(10), probabilities[i])
        axes[i, 1].set_xticks(range(10))
        axes[i, 1].set_xlabel('数字')
        axes[i, 1].set_ylabel('概率')
        axes[i, 1].set_title(f'预测概率分布')
    
    plt.tight_layout()
    plt.show()
    print("演示完成")

# 主函数
def main():
    parser = argparse.ArgumentParser(description='MNIST模型可视化和演示工具')
    parser.add_argument('--mode', type=str, default='history', 
                        choices=['history', 'predictions', 'features', 'demo'],
                        help='可视化模式')
    parser.add_argument('--model_path', type=str, default='models/best_model.pth',
                        help='训练好的模型权重文件路径')
    parser.add_argument('--model_type', type=str, default='medium',
                        choices=['simple', 'medium', 'advanced'],
                        help='模型类型')
    parser.add_argument('--history_path', type=str, default='models/training_history.npy',
                        help='训练历史文件路径')
    parser.add_argument('--output_dir', type=str, default='visualizations',
                        help='可视化结果保存目录')
    parser.add_argument('--num_samples', type=int, default=10,
                        help='可视化的样本数量')
    parser.add_argument('--layer_indices', type=int, nargs='+', default=[0, 2, 4],
                        help='要可视化的卷积层索引')
    
    args = parser.parse_args()
    set_seed()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.mode == 'history':
        # 可视化训练历史
        history = load_training_history(args.history_path)
        if history:
            save_path = os.path.join(args.output_dir, 'training_history.png')
            plot_training_history(history, save_path)
    
    elif args.mode == 'predictions':
        # 可视化预测结果
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = get_model(args.model_type).to(device)
        model.load_state_dict(torch.load(args.model_path, map_location=device))
       # 加载测试数据（使用config字典）
        data_config = {
            'batch_size': args.num_samples,
            'num_workers': 0,
            'pin_memory': False,
            'mean': (0.1307,),
            'std': (0.3081,),
            'data_dir': 'data'
        }
        data_loader = MNISTDataLoader(data_config)
        test_loader = data_loader.get_test_loader()
        
        save_path = os.path.join(args.output_dir, 'predictions.png')
        visualize_predictions(model, test_loader, device, args.num_samples, save_path)
    
    elif args.mode == 'features':
        # 可视化特征图
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = get_model(args.model_type).to(device)
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        
        # 加载测试图像（使用config字典）
        data_config = {
            'batch_size': 1,
            'num_workers': 0,
            'pin_memory': False,
            'mean': (0.1307,),
            'std': (0.3081,),
            'data_dir': 'data'
        }
        data_loader = MNISTDataLoader(data_config)
        test_loader = data_loader.get_test_loader()
        data_iter = iter(test_loader)
        image, _ = next(data_iter)
        image = image[0].to(device)
        
        save_path = os.path.join(args.output_dir, 'feature_maps')
        visualize_feature_maps(model, image, args.layer_indices, save_path)
    
    elif args.mode == 'demo':
        # 运行基于matplotlib的演示应用（替代streamlit）
        print("启动matplotlib演示应用...")
        run_matplotlib_demo(args.model_path, args.model_type)
        print("演示应用已运行完成")

if __name__ == '__main__':
    main()
