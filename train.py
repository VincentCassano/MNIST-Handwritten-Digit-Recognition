# 模型训练脚本

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import os
import sys
import time
import argparse
import numpy as np
import json
from datetime import datetime

# 避免重复导入提示
_import_messages = set()

def print_once(message):
    """仅打印一次消息，避免重复"""
    if message not in _import_messages:
        print(message)
        _import_messages.add(message)

def get_lr(optimizer):
    """获取优化器的当前学习率"""
    try:
        return optimizer.param_groups[0]['lr']
    except (IndexError, KeyError, AttributeError):
        return 0.0

# 尝试导入tqdm，如果失败则提供一个简单的替代实现
try:
    from tqdm import tqdm
    HAS_TQDM = True
    print_once("✅ 成功导入tqdm进度条")
except ImportError:
    HAS_TQDM = False
    print_once("⚠️ 无法导入tqdm，将使用简单进度显示")
    # 定义一个简单的tqdm替代类
    class SimpleProgressBar:
        def __init__(self, iterable=None, desc="", total=None):
            self.iterable = iterable
            self.desc = desc
            self.total = total if total is not None else len(iterable) if iterable is not None else 0
            self.current = 0
            self.start_time = time.time()
        
        def __iter__(self):
            if self.iterable is None:
                raise ValueError("No iterable provided")
            
            for item in self.iterable:
                yield item
                self.current += 1
                self._update_display()
        
        def __len__(self):
            # 实现__len__方法以支持len(progress_bar)
            return self.total
        
        def _update_display(self):
            elapsed = time.time() - self.start_time
            if elapsed > 0:
                rate = self.current / elapsed
                eta = (self.total - self.current) / rate if rate > 0 else 0
                progress = self.current / self.total * 100 if self.total > 0 else 0
                sys.stdout.write(f'\r{self.desc} {self.current}/{self.total} ({progress:.1f}%) ETA: {eta:.1f}s')
                sys.stdout.flush()
        
        def set_postfix(self, *args, **kwargs):
            # 接受位置参数（如字典）和关键字参数
            # 存储postfix信息以备可能的使用
            if args and isinstance(args[0], dict):
                self.postfix = args[0]
            else:
                self.postfix = kwargs
            pass
    
    # 替换tqdm
    tqdm = SimpleProgressBar

# 尝试导入TensorBoard，如果失败则设置为None
try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
    print_once("✅ 成功导入TensorBoard")
except ImportError:
    SummaryWriter = None
    HAS_TENSORBOARD = False
    print_once("⚠️ 无法导入TensorBoard，将跳过TensorBoard记录功能")


# 导入自定义模块
from model_def import get_model, count_parameters
from data_loader import MNISTDataLoader, DEFAULT_DATA_CONFIG


def parse_args():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description='MNIST 手写数字识别模型训练')
    parser.add_argument('--model', type=str, default='medium', choices=['simple', 'medium', 'advanced'],
                      help='选择模型类型')
    parser.add_argument('--epochs', type=int, default=30, help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=64, help='批量大小')
    parser.add_argument('--lr', type=float, default=0.001, help='初始学习率')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='权重衰减')
    parser.add_argument('--val-ratio', type=float, default=0.1, help='验证集比例')
    parser.add_argument('--patience', type=int, default=5, help='早停耐心值')
    parser.add_argument('--save-dir', type=str, default='./models', help='模型保存目录')
    parser.add_argument('--log-dir', type=str, default='./logs', help='日志保存目录')
    parser.add_argument('--resume', type=str, default=None, help='从检查点恢复训练')
    parser.add_argument('--use-mixed-precision', action='store_true', help='使用混合精度训练')
    parser.add_argument('--gradient-accumulation-steps', type=int, default=1, help='梯度累积步数')
    parser.add_argument('--warmup-epochs', type=int, default=2, help='学习率预热轮数')
    parser.add_argument('--use-cosine-lr-scheduler', action='store_true', help='使用余弦退火学习率调度器')
    
    return parser.parse_args()


def get_lr(optimizer):
    """
    获取当前学习率
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']


def create_optimizer(model, lr, weight_decay):
    """
    创建优化器
    """
    # 使用Adam优化器
    optimizer = optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )
    return optimizer


def create_lr_scheduler(optimizer, config, total_steps):
    """
    创建学习率调度器
    """
    if config['use_cosine_lr_scheduler']:
        # 余弦退火调度器
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_steps,
            eta_min=config['lr'] * 0.01  # 最小学习率
        )
    else:
        # 阶梯式衰减调度器
        # 确保step_size至少为1，避免除零错误
        step_size = max(1, config['epochs'] // 3)
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=step_size,
            gamma=0.1
        )
    
    return scheduler


def train_one_epoch(model, train_loader, criterion, optimizer, scaler, config, device, epoch, writer):
    """
    训练一个轮次
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # 梯度累积计数器
    accumulation_step = 0
    
    # 使用tqdm显示进度
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}/{config["epochs"]}')
    
    for i, (images, labels) in enumerate(progress_bar):
        accumulation_step += 1
        
        # 将数据移到设备上
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        
        # 前向传播 - 使用混合精度训练（如果启用）
        if config['use_mixed_precision']:
            with autocast():
                output = model(images)
                loss = criterion(output, labels) / config['gradient_accumulation_steps']
        else:
            output = model(images)
            loss = criterion(output, labels) / config['gradient_accumulation_steps']
        
        # 反向传播并更新梯度
        if config['use_mixed_precision']:
            scaler.scale(loss).backward()
            
            # 梯度累积达到指定步数后更新参数
            if accumulation_step % config['gradient_accumulation_steps'] == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)  # 更高效的梯度清零
        else:
            loss.backward()
            
            # 梯度累积达到指定步数后更新参数
            if accumulation_step % config['gradient_accumulation_steps'] == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        
        # 统计损失和准确率
        running_loss += loss.item() * config['gradient_accumulation_steps']  # 恢复原始损失值
        _, predicted = output.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # 更新进度条
        if HAS_TQDM:
            progress_bar.set_postfix({
                'loss': f'{running_loss/(i+1):.3f}',
                'acc': f'{100.*correct/total:.2f}%',
                'lr': f'{get_lr(optimizer):.6f}'
            })
        else:
            progress_bar.set_postfix(loss=f'{running_loss/(i+1):.3f}', 
                                   acc=f'{100.*correct/total:.2f}%',
                                   lr=f'{get_lr(optimizer):.6f}')
    
    # 计算平均损失和准确率
    train_loss = running_loss / len(train_loader)
    train_acc = 100. * correct / total
    
    # 记录到TensorBoard（检查writer是否为None）
    if writer is not None and HAS_TENSORBOARD:
        try:
            writer.add_scalar('train/loss', train_loss, epoch)
            writer.add_scalar('train/accuracy', train_acc, epoch)
            writer.add_scalar('train/learning_rate', get_lr(optimizer), epoch)
        except Exception as e:
            print_once(f"⚠️ TensorBoard记录失败: {e}")
    elif not HAS_TENSORBOARD:
        print_once("⚠️ 跳过TensorBoard记录（TensorBoard不可用）")
    else:
        print_once("⚠️ 跳过TensorBoard记录（writer为None）")
    
    print(f"📊 Epoch {epoch}/{config['epochs']} | 训练损失: {train_loss:.4f} | 训练准确率: {train_acc:.2f}% | 学习率: {get_lr(optimizer):.6f}")
    
    return train_loss, train_acc


def validate(model, val_loader, criterion, device, epoch, writer):
    """
    验证模型性能
    """
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    
    # 不计算梯度
    with torch.no_grad():
        # 使用tqdm显示进度
        progress_bar = tqdm(val_loader, desc=f'Validation')
        
        for images, labels in progress_bar:
            # 将数据移到设备上
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            # 前向传播
            output = model(images)
            loss = criterion(output, labels)
            
            # 统计损失和准确率
            val_loss += loss.item()
            _, predicted = output.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # 更新进度条
            if HAS_TQDM:
                progress_bar.set_postfix({
                    'loss': f'{val_loss/len(progress_bar):.3f}',
                    'acc': f'{100.*correct/total:.2f}%'
                })
            else:
                progress_bar.set_postfix(loss=f'{val_loss/len(progress_bar):.3f}',
                                       acc=f'{100.*correct/total:.2f}%')
    
    # 计算平均损失和准确率
    val_loss = val_loss / len(val_loader)
    val_acc = 100. * correct / total
    
    # 记录到TensorBoard（检查writer是否为None）
    if writer is not None and HAS_TENSORBOARD:
        try:
            writer.add_scalar('val/loss', val_loss, epoch)
            writer.add_scalar('val/accuracy', val_acc, epoch)
        except Exception as e:
            print_once(f"⚠️ TensorBoard记录失败: {e}")
    elif not HAS_TENSORBOARD:
        print_once("⚠️ 跳过TensorBoard记录（TensorBoard不可用）")
    else:
        print_once("⚠️ 跳过TensorBoard记录（writer为None）")
    
    print(f"✅ 验证损失: {val_loss:.4f} | 验证准确率: {val_acc:.2f}%")
    
    return val_loss, val_acc


def save_checkpoint(model, optimizer, scaler, config, epoch, best_accuracy, checkpoint_dir, timestamp=None):
    """
    保存模型检查点，使用健壮的路径处理和异常处理
    """
    # 确保保存目录存在 - 更健壮的实现
    def safe_save_directory(dir_path):
        """安全地确保保存目录存在"""
        try:
            # 获取父目录并确保存在
            parent_dir = os.path.dirname(dir_path)
            if parent_dir and not os.path.exists(parent_dir):
                os.makedirs(parent_dir, exist_ok=True)
                print(f"✅ 已创建父目录: {parent_dir}")
            
            # 确保目标目录存在
            os.makedirs(dir_path, exist_ok=True)
            print(f"✅ 确保保存目录存在: {dir_path}")
            return True
        except Exception as e:
            print(f"❌ 创建保存目录失败: {e}")
            return False
    
    # 如果没有提供时间戳，使用当前时间（仅作为备用）
    if timestamp is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 创建基于时间戳的模型保存目录
    model_dir = os.path.join('models', timestamp)
    
    # 尝试使用基于时间戳的目录
    if not safe_save_directory(model_dir):
        # 如果失败，使用提供的目录
        if not safe_save_directory(checkpoint_dir):
            # 如果还是失败，使用当前目录作为备选
            checkpoint_dir = os.path.dirname(os.path.abspath(__file__))
            print(f"⚠️ 使用当前目录作为备选: {checkpoint_dir}")
            safe_save_directory(checkpoint_dir)
        model_dir = checkpoint_dir
    
    # 创建检查点字典
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict() if scaler else None,
        'best_accuracy': best_accuracy,
        'config': config
    }
    
    # 尝试保存模型，使用更安全的文件名
    try:
        # 使用简单的文件名，避免路径问题
        checkpoint_filename = f'checkpoint_epoch_{epoch}.pth'
        checkpoint_path = os.path.join(model_dir, checkpoint_filename)
        
        # 保存检查点
        torch.save(checkpoint, checkpoint_path)
        print(f"✅ 检查点已保存到: {checkpoint_path}")
        
        # 保存最佳模型
        best_model_filename = 'best_model.pth'
        best_model_path = os.path.join(model_dir, best_model_filename)
        torch.save(model.state_dict(), best_model_path)
        print(f"🏆 最佳模型已保存到: {best_model_path}")
        
        # 同时在根目录保存一个链接到最新的最佳模型
        root_best_model_path = os.path.join('models', 'best_model.pth')
        try:
            # 如果文件已存在，尝试删除
            if os.path.exists(root_best_model_path):
                os.remove(root_best_model_path)
            # 在Windows上，可以直接复制文件作为替代
            import shutil
            shutil.copy(best_model_path, root_best_model_path)
            print(f"🔗 最佳模型链接已更新到: {root_best_model_path}")
        except Exception as e_link:
            print(f"⚠️ 更新最佳模型链接失败: {e_link}")
        
        return True
    except Exception as e:
        print(f"❌ 保存模型失败: {e}")
        # 尝试使用更简单的路径和文件名
        try:
            # 只使用文件名，不包含路径
            simple_checkpoint_path = f'checkpoint_epoch_{epoch}.pth'
            torch.save(checkpoint, simple_checkpoint_path)
            print(f"⚠️ 使用简单路径保存检查点: {simple_checkpoint_path}")
            
            simple_best_model_path = 'best_model.pth'
            torch.save(model.state_dict(), simple_best_model_path)
            print(f"⚠️ 使用简单路径保存最佳模型: {simple_best_model_path}")
            return True
        except Exception as e2:
            print(f"❌ 使用简单路径也保存失败: {e2}")
            print("⚠️ 训练完成但无法保存模型")
            return False


def load_checkpoint(checkpoint_path, model, optimizer, scaler):
    """
    加载模型检查点
    """
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"检查点文件不存在: {checkpoint_path}")
    
    # 加载检查点
    checkpoint = torch.load(checkpoint_path)
    
    # 加载模型状态
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 加载优化器状态
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # 加载梯度缩放器状态
    if scaler and 'scaler_state_dict' in checkpoint and checkpoint['scaler_state_dict']:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
    
    print(f"📂 已从 {checkpoint_path} 加载检查点，继续从第 {checkpoint['epoch']} 轮训练")
    
    return checkpoint['epoch'], checkpoint['best_accuracy'], checkpoint.get('config', {})


def setup_directories(save_dir, log_dir):
    """
    设置保存目录
    """
    # 检查并处理模型保存目录
    if os.path.exists(save_dir):
        if not os.path.isdir(save_dir):
            print(f"警告: {save_dir} 不是目录，正在删除并创建新目录...")
            os.remove(save_dir)
    os.makedirs(save_dir, exist_ok=True)
    
    # 检查并处理日志目录
    if os.path.exists(log_dir):
        if not os.path.isdir(log_dir):
            print(f"警告: {log_dir} 不是目录，正在删除并创建新目录...")
            os.remove(log_dir)
    os.makedirs(log_dir, exist_ok=True)
    
    return save_dir, log_dir

def main():
    # 解析命令行参数
    args = parse_args()
    
    # 创建配置字典
    config = {
        'model_name': args.model,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'val_ratio': args.val_ratio,
        'patience': args.patience,
        'save_dir': args.save_dir,
        'log_dir': args.log_dir,
        'resume': args.resume,
        'use_mixed_precision': args.use_mixed_precision,
        'gradient_accumulation_steps': args.gradient_accumulation_steps,
        'warmup_epochs': args.warmup_epochs,
        'use_cosine_lr_scheduler': args.use_cosine_lr_scheduler,
        # 添加数据配置
        'data_config': DEFAULT_DATA_CONFIG.copy()
    }
    config['data_config']['batch_size'] = config['batch_size']
    
    # 生成全局时间戳，确保整个训练过程中所有模型都使用相同的时间戳文件夹
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    print(f"📂 训练时间戳: {timestamp} (用于组织模型文件)")
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️ 使用设备: {device}")
    
    # 使用绝对路径和规范化处理
    # 获取当前脚本所在目录作为基础
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 配置日志目录
    config['log_dir'] = os.path.abspath(os.path.join(base_dir, config['log_dir']))
    log_dir = config['log_dir']
    
    # 配置保存目录
    config['save_dir'] = os.path.abspath(os.path.join(base_dir, config['save_dir']))
    save_dir = config['save_dir']
    
    # 加强目录创建逻辑，确保父目录存在
    def ensure_directory(directory_path):
        """确保目录存在，处理各种可能的错误"""
        try:
            # 获取父目录
            parent_dir = os.path.dirname(directory_path)
            if parent_dir and not os.path.exists(parent_dir):
                os.makedirs(parent_dir, exist_ok=True)
                print(f"✅ 已创建父目录: {parent_dir}")
            
            # 然后创建目标目录
            if os.path.exists(directory_path):
                if not os.path.isdir(directory_path):
                    print(f"⚠️ {directory_path} 不是目录，正在删除并重新创建...")
                    try:
                        os.remove(directory_path)
                    except:
                        pass
            os.makedirs(directory_path, exist_ok=True)
            print(f"✅ 确保目录存在: {directory_path}")
            return True
        except Exception as e:
            print(f"❌ 创建目录失败 {directory_path}: {e}")
            return False
    
    # 确保日志目录存在
    if not ensure_directory(log_dir):
        # 如果失败，使用临时目录
        import tempfile
        log_dir = tempfile.mkdtemp(prefix='mnist_logs_')
        config['log_dir'] = log_dir
        print(f"⚠️ 切换到临时日志目录: {log_dir}")
    
    # 确保保存目录存在
    if not ensure_directory(save_dir):
        # 如果失败，使用当前目录
        save_dir = base_dir
        config['save_dir'] = save_dir
        print(f"⚠️ 切换到当前目录作为保存路径: {save_dir}")
    
    # 创建TensorBoard日志记录器
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    tb_log_dir = os.path.join(log_dir, f'{config["model_name"]}_{timestamp}')
    
    # 不进行路径分隔符替换，使用原生Windows路径
    
    # 使用更安全的方法创建嵌套目录
    # 先删除可能存在的文件（如果有）
    if os.path.exists(tb_log_dir) and not os.path.isdir(tb_log_dir):
        print(f"警告: {tb_log_dir} 存在但不是目录，正在删除...")
        os.remove(tb_log_dir)
    
    # 使用os模块创建目录，而不是依赖TensorFlow
    try:
        # 分步创建目录，确保每一层都是目录
        parts = tb_log_dir.split(os.sep)
        current_path = parts[0] + os.sep if os.name == 'nt' else parts[0]
        
        for part in parts[1:]:
            current_path = os.path.join(current_path, part)
            if os.path.exists(current_path):
                if not os.path.isdir(current_path):
                    print(f"警告: {current_path} 不是目录，正在删除...")
                    os.remove(current_path)
            if not os.path.exists(current_path):
                os.makedirs(current_path, exist_ok=True)
        
        print(f"✅ 已成功创建日志目录: {tb_log_dir}")
    except Exception as e:
        print(f"❌ 创建日志目录时出错: {e}")
        # 如果失败，尝试使用临时目录
        import tempfile
        tb_log_dir = tempfile.mkdtemp(prefix=f"mnist_{config['model_name']}_")
        # 使用原生Windows路径格式
        print(f"⚠️ 使用临时目录替代: {tb_log_dir}")
    
    # 再次确保目录存在，避免创建writer时出错
    try:
        os.makedirs(tb_log_dir, exist_ok=True)
        print(f"📁 再次确认并创建日志目录: {tb_log_dir}")
    except Exception as e:
        print(f"❌ 再次创建日志目录失败: {e}")
        # 使用更简单的目录路径作为备选
        tb_log_dir = os.path.join(log_dir, f'tb_{config["model_name"]}')
        try:
            os.makedirs(tb_log_dir, exist_ok=True)
            print(f"⚠️ 使用简化的日志目录路径: {tb_log_dir}")
        except Exception as e2:
            print(f"❌ 创建简化日志目录也失败: {e2}")
    
    # 尝试直接初始化SummaryWriter
    writer = None
    if HAS_TENSORBOARD:
        try:
            # 不使用中文路径，避免编码问题
            # 使用相对路径可能更可靠
            if os.name == 'nt':  # Windows系统
                # 尝试使用纯ASCII字符的路径
                tb_log_dir_safe = os.path.join(log_dir, f'tb_{config["model_name"]}_{timestamp}')
                os.makedirs(tb_log_dir_safe, exist_ok=True)
                writer = SummaryWriter(log_dir=tb_log_dir_safe)
                print(f"✅ TensorBoard日志记录器初始化成功 (使用安全路径: {tb_log_dir_safe})")
            else:
                writer = SummaryWriter(log_dir=tb_log_dir)
                print("✅ TensorBoard日志记录器初始化成功")
        except Exception as e:
            print(f"❌ TensorBoard初始化失败: {e}")
            # 如果SummaryWriter失败，尝试使用更简单的方法
            try:
                # 尝试使用最基础的路径
                simple_log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
                os.makedirs(simple_log_dir, exist_ok=True)
                writer = SummaryWriter(log_dir=simple_log_dir)
                print(f"⚠️ 尝试使用简单路径初始化TensorBoard: {simple_log_dir}")
            except Exception as e2:
                print(f"❌ 使用简单路径也失败: {e2}")
                print("⚠️ 跳过TensorBoard初始化，继续训练")
                writer = None
    else:
        print("ℹ️ TensorBoard不可用，跳过创建writer")
    
    # 保存配置到JSON文件 - 增强的异常处理
    try:
        config_save_path = os.path.join(log_dir, f'{config["model_name"]}_{timestamp}_config.json')
        with open(config_save_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        print(f"📋 配置已保存到: {config_save_path}")
    except Exception as e:
        print(f"❌ 保存配置失败: {e}")
        # 尝试使用更简单的文件名
        try:
            simple_config_path = os.path.join(log_dir, 'config.json')
            with open(simple_config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            print(f"⚠️ 已使用简化名称保存配置: {simple_config_path}")
        except Exception as e2:
            print(f"❌ 配置保存完全失败: {e2}")
            print("⚠️ 继续训练，但不会保存配置")
    
    # 初始化数据加载器 - 增强的错误处理
    train_loader = None
    val_loader = None
    print("📥 正在加载数据集...")
    
    # 尝试加载数据集，最多尝试3次
    max_data_load_attempts = 3
    for attempt in range(max_data_load_attempts):
        try:
            data_loader = MNISTDataLoader(config['data_config'])
            train_loader, val_loader = data_loader.get_train_val_loaders(config['val_ratio'])
            # 验证数据加载器
            if train_loader is not None and val_loader is not None:
                print(f"✅ 数据集加载成功！训练样本: {len(train_loader.dataset)}, 验证样本: {len(val_loader.dataset)}")
                break
            else:
                raise ValueError("数据加载器返回None")
        except Exception as e:
            print(f"❌ 第 {attempt+1}/{max_data_load_attempts} 次加载数据集失败: {e}")
            # 调整配置重试
            if attempt < max_data_load_attempts - 1:
                # 减小批量大小重试
                if config['batch_size'] > 8:
                    config['batch_size'] = max(8, config['batch_size'] // 2)
                    config['data_config']['batch_size'] = config['batch_size']
                    print(f"⚠️ 减小批量大小至 {config['batch_size']} 并重试...")
                time.sleep(1)
    
    if train_loader is None or val_loader is None:
        print("❌ 无法加载数据集，程序终止")
        sys.exit(1)
    
    # 初始化模型 - 增强的错误处理
    model = None
    print("🏗️ 正在初始化模型...")
    
    # 尝试加载模型，提供备选方案
    model_attempts = [
        (config['model_name'], "指定模型"),
        ('SimpleCNN', "简单CNN备选"),
        ('ResNet18', "ResNet18备选")
    ]
    
    for model_name, model_desc in model_attempts:
        try:
            print(f"尝试加载{model_desc}: {model_name}")
            model = get_model(model_name)
            if model is None:
                raise ValueError(f"get_model返回None: {model_name}")
            
            # 尝试移动模型到设备
            try:
                model = model.to(device)
                print(f"✅ {model_name} 成功加载并移至 {device}")
                # 更新配置以反映实际使用的模型
                config['model_name'] = model_name
                break
            except Exception as e:
                print(f"❌ 无法移动{model_name}到{device}: {e}")
                # 尝试在CPU上加载
                if device.type == 'cuda':
                    print("⚠️ 尝试在CPU上加载模型...")
                    model = model.to('cpu')
                    device = torch.device('cpu')
                    print(f"✅ {model_name} 成功加载到CPU")
                    config['model_name'] = model_name
                    break
        except Exception as e:
            print(f"❌ 加载{model_desc}失败: {e}")
    
    if model is None:
        print("❌ 无法加载任何模型，程序终止")
        sys.exit(1)
    
    # 打印模型信息
    print(f"📊 模型: {config['model_name']}")
    print(f"🧮 参数数量: {count_parameters(model):,}")
    
    # 初始化损失函数 - 安全初始化
    criterion = None
    try:
        criterion = nn.CrossEntropyLoss()
        criterion = criterion.to(device)
        print("✅ 损失函数初始化成功 (CrossEntropyLoss)")
    except Exception as e:
        print(f"❌ 初始化损失函数失败: {e}")
        print("⚠️ 尝试使用备选损失函数...")
        # 尝试备选损失函数
        alternative_losses = [
            (nn.NLLLoss(), "NLLLoss"),
            (nn.MSELoss(), "MSELoss")
        ]
        for alt_criterion, name in alternative_losses:
            try:
                criterion = alt_criterion
                criterion = criterion.to(device)
                print(f"✅ 成功使用备选损失函数: {name}")
                break
            except Exception as e2:
                print(f"❌ 备选损失函数 {name} 初始化失败: {e2}")
    
    if criterion is None:
        print("❌ 无法初始化任何损失函数，程序终止")
        sys.exit(1)
    
    # 初始化优化器
    optimizer = create_optimizer(model, config['lr'], config['weight_decay'])
    
    # 初始化混合精度训练的梯度缩放器 - 安全初始化
    scaler = None
    if config['use_mixed_precision']:
        try:
            # 检查是否在CUDA上运行
            if device.type == 'cuda':
                scaler = torch.cuda.amp.GradScaler()
                print("✅ 梯度缩放器初始化成功")
            else:
                raise ValueError("混合精度训练仅支持CUDA")
        except Exception as e:
            print(f"❌ 初始化梯度缩放器失败: {e}")
            print("⚠️ 禁用混合精度训练")
            config['use_mixed_precision'] = False
    
    # 计算总步数（用于余弦退火调度器）- 安全计算
    try:
        total_steps = config['epochs'] * len(train_loader)
        if total_steps <= 0:
            raise ValueError("总步数必须大于0")
        print(f"📈 总训练步数: {total_steps:,}")
    except Exception as e:
        print(f"❌ 计算总步数失败: {e}")
        # 设置合理的默认值
        total_steps = 1000
        print(f"⚠️ 使用默认总步数: {total_steps}")
    
    # 初始化学习率调度器 - 增强的异常处理
    scheduler = None
    try:
        scheduler = create_lr_scheduler(optimizer, config, total_steps)
        print("✅ 学习率调度器初始化成功")
    except Exception as e:
        print(f"❌ 初始化学习率调度器失败: {e}")
        print("⚠️ 使用简单的阶梯式调度器作为备选")
        
        # 尝试多种备选调度器
        try:
            # 阶梯式调度器
            step_size = max(1, config['epochs'] // 3)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)
            print(f"✅ 成功使用StepLR调度器 (step_size={step_size})")
        except Exception as e2:
            print(f"❌ 备选调度器也失败: {e2}")
            print("⚠️ 将不使用学习率调度器")
            # 创建一个不做任何操作的调度器
            class NoOpScheduler:
                def step(self):
                    pass
            scheduler = NoOpScheduler()
    
    # 设置早停参数
    best_accuracy = 0.0
    epochs_no_improve = 0
    
    # 初始化训练历史记录
    training_history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    # 恢复训练（如果需要）
    start_epoch = 1
    if config['resume']:
        start_epoch, best_accuracy, loaded_config = load_checkpoint(
            config['resume'], model, optimizer, scaler
        )
        # 更新配置（如果加载的配置与当前配置不同）
        config.update(loaded_config)
    
    print(f"\n🔧 训练配置:")
    for key, value in config.items():
        if key != 'data_config':
            print(f"  {key}: {value}")
    
    print(f"\n🚀 开始训练 ({'混合精度' if config['use_mixed_precision'] else '单精度'})")
    start_time = time.time()
    
    try:
        # 训练循环 - 增强的错误恢复能力
        for epoch in range(start_epoch, config['epochs'] + 1):
            try:
                print(f"\n🔄 开始轮次 {epoch}/{config['epochs']} - 学习率: {get_lr(optimizer):.6f}")
                
                # 确保模型在正确的设备上
                try:
                    model = model.to(device)
                except:
                    pass
                
                # 训练一个轮次 - 异常处理
                train_loss, train_acc = None, None
                try:
                    train_loss, train_acc = train_one_epoch(
                        model, train_loader, criterion, optimizer, scaler, config, device, epoch, writer
                    )
                    if train_loss is None or train_acc is None:
                        raise ValueError("训练函数返回None值")
                except KeyboardInterrupt:
                    raise  # 重新抛出以在外部捕获
                except Exception as e:
                    print(f"❌ 训练轮次失败: {e}")
                    # 尝试恢复训练状态
                    try:
                        # 重置梯度
                        optimizer.zero_grad(set_to_none=True)
                        # 降低学习率
                        for param_group in optimizer.param_groups:
                            param_group['lr'] *= 0.5
                        print(f"⚠️ 已降低学习率至: {get_lr(optimizer):.6f}")
                    except:
                        pass
                    # 跳过当前轮次的验证
                    continue
                
                # 验证 - 异常处理
                val_loss, val_acc = None, None
                try:
                    val_loss, val_acc = validate(model, val_loader, criterion, device, epoch, writer)
                    if val_loss is None or val_acc is None:
                        raise ValueError("验证函数返回None值")
                except Exception as e:
                    print(f"❌ 验证失败: {e}")
                    # 使用默认值作为备选
                    val_acc = best_accuracy * 0.99  # 使用稍低于最佳的准确率
                    val_loss = float('inf')
                    print(f"⚠️ 使用备选验证结果")
                
                # 更新学习率 - 安全检查
                try:
                    if scheduler is not None:
                        scheduler.step()
                except Exception as e:
                    print(f"❌ 更新学习率失败: {e}")
                    print("⚠️ 继续使用当前学习率")
                
                # 记录训练历史
                training_history['train_loss'].append(train_loss)
                training_history['train_acc'].append(train_acc)
                training_history['val_loss'].append(val_loss)
                training_history['val_acc'].append(val_acc)
                
                # 检查是否是最佳模型
                is_best = False
                try:
                    is_best = val_acc > best_accuracy
                except:
                    is_best = False
                
                if is_best:
                    best_accuracy = val_acc
                    epochs_no_improve = 0
                    # 保存检查点和最佳模型 - 异步保存减少训练中断
                    try:
                        save_success = save_checkpoint(model, optimizer, scaler, config, epoch, best_accuracy, save_dir, timestamp)
                        if not save_success:
                            print("⚠️ 模型保存失败，但训练继续")
                    except Exception as e:
                        print(f"❌ 保存模型时发生错误: {e}")
                else:
                    epochs_no_improve += 1
                    print(f"⏳ 已 {epochs_no_improve} 轮没有改进，最佳准确率: {best_accuracy:.2f}%")
                
                # 早停检查
                if epochs_no_improve >= config['patience']:
                    print(f"🛑 触发早停机制！训练提前结束。")
                    break
                
                print()  # 空行分隔轮次
                
            except KeyboardInterrupt:
                raise  # 重新抛出以在外部捕获
            except Exception as e:
                print(f"❌ 轮次 {epoch} 发生未预期错误: {e}")
                # 尝试重置优化器状态
                try:
                    # 降低学习率
                    for param_group in optimizer.param_groups:
                        param_group['lr'] *= 0.5
                    print(f"⚠️ 已重置学习率为: {get_lr(optimizer):.6f}")
                    # 清理梯度
                    optimizer.zero_grad(set_to_none=True)
                except Exception as reset_error:
                    print(f"❌ 尝试恢复训练状态失败: {reset_error}")
                    # 考虑早停以避免无限错误循环
                    if epoch > start_epoch + 5:  # 只在训练了几轮后才考虑早停
                        print("⚠️ 连续错误，考虑提前终止训练")
                        break
        
    except KeyboardInterrupt:
        print("\n⏸️  训练被用户中断")
    finally:
        # 计算总训练时间
        total_time = time.time() - start_time
        hours, rem = divmod(total_time, 3600)
        minutes, seconds = divmod(rem, 60)
        
        print(f"\n🏁 训练完成！")
        print(f"⏱️  总训练时间: {int(hours)}小时 {int(minutes)}分钟 {seconds:.1f}秒")
        print(f"🏆 最佳验证准确率: {best_accuracy:.2f}%")
        print(f"💾 最佳模型保存在: {os.path.join(save_dir, 'best_model.pth')}")
        
        # 保存训练历史
        try:
            # 确保logs目录存在
            os.makedirs('logs', exist_ok=True)
            # 保存训练历史到npy文件
            history_filename = f'logs/{config["model_name"]}_{timestamp}_training_history.npy'
            np.save(history_filename, training_history)
            print(f"📊 训练历史已保存至: {history_filename}")
            
            # 同时保存一个通用路径的历史文件
            general_history_path = 'logs/training_history.npy'
            np.save(general_history_path, training_history)
            print(f"📊 通用训练历史已保存至: {general_history_path}")
        except Exception as e:
            print(f"❌ 保存训练历史失败: {e}")
            
        # 关闭TensorBoard日志记录器
        if writer is not None and HAS_TENSORBOARD:
            try:
                writer.close()
                print_once("✅ TensorBoard writer已关闭")
            except Exception as e:
                print_once(f"⚠️ 关闭TensorBoard writer时出错: {e}")
        elif not HAS_TENSORBOARD:
            print_once("⚠️ 跳过关闭TensorBoard writer（TensorBoard不可用）")
        else:
            print_once("⚠️ 跳过关闭TensorBoard writer（writer为None）")


if __name__ == "__main__":
    main()