# 模型评估和测试脚本

import os
import torch
import numpy as np
import argparse
# 移除对scikit-learn和seaborn的依赖，使用纯PyTorch和NumPy实现核心功能

from model_def import get_model
from data_loader import MNISTDataLoader

# 设置随机种子以确保结果可复现
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 评估模型在测试集上的性能
def evaluate_model(model, test_loader, device):
    model.eval()  # 设置模型为评估模式
    all_preds = []
    all_labels = []
    all_inputs = []  # 保存输入图像以便后续可视化
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # 保存预测结果、真实标签和输入图像
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_inputs.extend(inputs.cpu())
    
    accuracy = 100 * correct / total
    print(f'测试集准确率: {accuracy:.2f}%')
    
    return np.array(all_preds), np.array(all_labels), torch.stack(all_inputs), accuracy

# 使用NumPy计算混淆矩阵
def compute_confusion_matrix(y_true, y_pred, num_classes=10):
    """使用纯NumPy计算混淆矩阵"""
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm

# 打印混淆矩阵
def print_confusion_matrix(cm):
    """打印混淆矩阵"""
    print("\n混淆矩阵:")
    # 打印表头
    print("    ", end="")
    for i in range(cm.shape[1]):
        print(f"{i:4d}", end="")
    print()
    print("    " + "----" * cm.shape[1])
    
    # 打印每一行
    for i in range(cm.shape[0]):
        print(f"{i:2d} |", end="")
        for j in range(cm.shape[1]):
            print(f"{cm[i, j]:4d}", end="")
        print()

# 生成简单的分类报告
def generate_classification_report(y_true, y_pred, num_classes=10):
    """使用纯NumPy生成简单的分类报告"""
    cm = compute_confusion_matrix(y_true, y_pred, num_classes)
    
    print("\n简单分类报告:")
    print(f"{'类别':<5} {'准确率':<10} {'召回率':<10} {'F1分数':<10}")
    print("-" * 40)
    
    total_precision = 0
    total_recall = 0
    total_f1 = 0
    count_classes = 0
    
    for i in range(num_classes):
        # 准确率 (precision) = 正确预测为i的样本 / 所有预测为i的样本
        precision = cm[i, i] / cm[:, i].sum() if cm[:, i].sum() > 0 else 0
        # 召回率 (recall) = 正确预测为i的样本 / 所有实际为i的样本
        recall = cm[i, i] / cm[i, :].sum() if cm[i, :].sum() > 0 else 0
        # F1分数
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"{i:<5} {precision:.4f}     {recall:.4f}     {f1:.4f}")
        
        # 计算平均值（只计算有样本的类别）
        if cm[i, :].sum() > 0:
            total_precision += precision
            total_recall += recall
            total_f1 += f1
            count_classes += 1
    
    # 打印平均值
    if count_classes > 0:
        print("-" * 40)
        print(f"{'平均':<5} {total_precision/count_classes:.4f}     {total_recall/count_classes:.4f}     {total_f1/count_classes:.4f}")
    
    return cm

# 找出错误分类的样本
def find_misclassified_samples(images, y_true, y_pred, num_samples=20):
    """找出错误分类的样本并打印信息"""
    # 找出所有错误分类的样本索引
    misclassified_indices = np.where(y_true != y_pred)[0]
    
    if len(misclassified_indices) == 0:
        print("没有错误分类的样本!")
        return misclassified_indices
    
    print(f"\n发现 {len(misclassified_indices)} 个错误分类的样本")
    
    # 选择有限数量的错误分类样本
    num_to_show = min(num_samples, len(misclassified_indices))
    selected_indices = np.random.choice(misclassified_indices, num_to_show, replace=False)
    
    print("前几个错误分类的样本:")
    for i, idx in enumerate(selected_indices[:10]):  # 只打印前10个
        print(f"样本 {idx}: 真实标签={y_true[idx]}, 预测标签={y_pred[idx]}")
    
    return misclassified_indices

# 分析每个类别的性能
def analyze_class_performance(cm):
    class_accuracy = []
    for i in range(10):  # MNIST有10个类别
        if cm[i, :].sum() > 0:
            acc = cm[i, i] / cm[i, :].sum()
            class_accuracy.append(acc)
            print(f'类别 {i} 的准确率: {acc:.4f}')
    
    avg_class_accuracy = sum(class_accuracy) / len(class_accuracy)
    print(f'平均类别准确率: {avg_class_accuracy:.4f}')
    
    return class_accuracy, avg_class_accuracy

# 主函数
def main():
    parser = argparse.ArgumentParser(description='MNIST模型评估脚本')
    # parser.add_argument('--model_path', type=str, default='models/best_model.pth',
    parser.add_argument('--model_path', type=str, default='best_model.pth',
                        help='训练好的模型权重文件路径')
    parser.add_argument('--model_type', type=str, default='medium', 
                        choices=['simple', 'medium', 'advanced'], 
                        help='模型类型')
    parser.add_argument('--batch_size', type=int, default=128, 
                        help='测试时的批次大小')
    parser.add_argument('--num_workers', type=int, default=4, 
                        help='数据加载器的工作线程数')
    parser.add_argument('--visualize', action='store_true', default=True, 
                        help='是否可视化结果')
    parser.add_argument('--output_dir', type=str, default='results', 
                        help='结果保存目录')
    
    args = parser.parse_args()
    set_seed()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 检查CUDA可用性
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    
    # 创建数据加载器配置
    data_config = {
        'batch_size': args.batch_size,
        'num_workers': args.num_workers,
        'pin_memory': True,
        'mean': (0.1307,),
        'std': (0.3081,),
        'data_dir': './data'
    }
    
    # 初始化数据加载器
    data_loader = MNISTDataLoader(data_config)
    test_loader = data_loader.get_test_loader()
    
    # 加载模型
    model = get_model(model_name=args.model_type).to(device)
    
    # 加载训练好的权重
    try:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print(f'成功加载模型权重: {args.model_path}')
    except Exception as e:
        print(f'加载模型权重失败: {e}')
        print('请确保指定的模型路径正确')
        return
    
    # 评估模型
    print('开始评估模型...')
    all_preds, all_labels, all_inputs, accuracy = evaluate_model(model, test_loader, device)
    
    # 生成分类报告并获取混淆矩阵
    cm = generate_classification_report(all_labels, all_preds)
    
    # 分析每个类别的性能
    class_accuracy, avg_class_accuracy = analyze_class_performance(cm)
    
    # 打印混淆矩阵
    print_confusion_matrix(cm)
    
    # 找出错误分类的样本
    if args.visualize:
        misclassified_indices = find_misclassified_samples(all_inputs, all_labels, all_preds)
    
    # 保存评估结果到文本文件（使用utf-8编码处理中文字符）
    result_file = os.path.join(args.output_dir, 'evaluation_results.txt')
    try:
        with open(result_file, 'w', encoding='utf-8') as f:
            f.write(f'模型评估结果\n')
            f.write(f'------------------------\n')
            f.write(f'模型类型: {args.model_type}\n')
            f.write(f'测试集准确率: {accuracy:.4f}\n\n')
            
            f.write('每个类别的准确率:\n')
            for i in range(10):
                if cm[i, :].sum() > 0:
                    acc = cm[i, i] / cm[i, :].sum()
                    f.write(f'类别 {i}: {acc:.4f}\n')
    except Exception as e:
        print(f'保存评估结果时出错: {e}')
    
    print(f'评估结果已保存至: {args.output_dir}/evaluation_results.txt')
    print('模型评估完成!')

if __name__ == '__main__':
    main()
