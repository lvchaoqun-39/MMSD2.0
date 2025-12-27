import os
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7897'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7897'
import argparse
import torch
import numpy as np
import random
# 修改导入语句，使用相对导入
from data_set import MyDataset
from model import MV_CLIP, RCLMFN
from train import train, test
from transformers import AdamW, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def set_args():
    parser = argparse.ArgumentParser(description='MMSD2.0模型训练和评估')
    
    # 基础配置
    parser.add_argument('--model_name', type=str, default='RCLMFN', choices=['MV_CLIP', 'RCLMFN'], 
                        help='模型名称')
    parser.add_argument('--dataset', type=str, default='MMSD2.0', help='数据集名称')
    parser.add_argument('--data_dir', type=str, default='data/text_json_final', help='数据目录')
    parser.add_argument('--image_dir', type=str, default='data/dataset_image', help='图像目录')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='模型保存目录')
    
    # 训练参数
    parser.add_argument('--num_epochs', type=int, default=10, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=16, help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='权重衰减')
    parser.add_argument('--warmup_ratio', type=float, default=0.1, help='预热比例')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='梯度裁剪阈值')
    parser.add_argument('--dropout_rate', type=float, default=0.1, help='dropout率')
    parser.add_argument('--save_epoch', type=int, default=5, help='每多少轮保存一次模型')
    parser.add_argument('--log_step', type=int, default=100, help='日志记录间隔')
    parser.add_argument('--use_wandb', type=bool, default=True, help='是否使用wandb')
    
    # 模型参数 - 共享
    parser.add_argument('--label_number', type=int, default=2, help='标签数量')
    
    # 模型参数 - MV_CLIP
    parser.add_argument('--text_size', type=int, default=512, help='文本特征维度')
    parser.add_argument('--image_size', type=int, default=768, help='图像特征维度')
    parser.add_argument('--layers', type=int, default=2, help='Transformer层数')
    parser.add_argument('--simple_linear', type=bool, default=False, help='是否使用简单线性层')
    
    # 模型参数 - RCLMFN
    parser.add_argument('--hidden_size', type=int, default=768, help='隐藏层维度')
    parser.add_argument('--num_heads', type=int, default=4, help='注意力头数量')
    parser.add_argument('--fusion_layers', type=int, default=2, help='融合层数')
    parser.add_argument('--max_seq_length', type=int, default=64, help='最大序列长度')
    
    # 优化器选择
    parser.add_argument('--optimizer', type=str, default='AdamW', choices=['AdamW', 'Adam'], help='优化器类型')
    parser.add_argument('--schedule', type=str, default='linear', choices=['linear', 'cosine'], help='学习率调度器')
    
    # 设备配置
    parser.add_argument('--device', type=str, default='cuda', help='设备选择')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    
    # 测试模式
    parser.add_argument('--test_only', type=bool, default=False, help='是否仅测试')
    parser.add_argument('--test_model_path', type=str, default='', help='测试模型路径')
    
    return parser.parse_args()

def seed_everything(seed):
    """
    设置随机种子，确保实验可复现
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) if torch.cuda.is_available() else None
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_optimizer_and_scheduler(args, model, train_data):
    """
    获取优化器和学习率调度器
    """
    # 区分参数组，对不同部分使用不同的学习率
    if args.model_name == 'RCLMFN':
        # RCLMFN模型的参数分组
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                'weight_decay': args.weight_decay,
            },
            {
                'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,
            }
        ]
        
        # 对于预训练模型部分使用较小的学习率
        pretrain_params = [p for n, p in model.named_parameters() 
                          if 'text_encoder' in n or 'image_encoder' in n]
        task_params = [p for n, p in model.named_parameters() 
                      if 'text_encoder' not in n and 'image_encoder' not in n]
        
        optimizer_grouped_parameters = [
            {'params': pretrain_params, 'lr': args.learning_rate * 0.1, 'weight_decay': args.weight_decay},
            {'params': task_params, 'lr': args.learning_rate, 'weight_decay': args.weight_decay}
        ]
    else:
        # MV_CLIP模型的参数分组
        optimizer_grouped_parameters = model.parameters()
    
    # 选择优化器
    if args.optimizer == 'AdamW':
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=1e-8)
    else:
        optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=args.learning_rate, eps=1e-8)
    
    # 计算总步数
    total_steps = len(train_data) // args.batch_size * args.num_epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    
    # 选择学习率调度器
    if args.schedule == 'linear':
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
    else:
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
    
    return optimizer, scheduler

def main():
    # 设置参数
    args = set_args()
    
    # 设置随机种子
    seed_everything(args.seed)
    
    # 设置设备
    # 处理数字形式的设备ID，将其格式化为'cuda:数字'
    device_arg = args.device
    if device_arg.isdigit():
        device_arg = f'cuda:{device_arg}'
    device = torch.device(device_arg if torch.cuda.is_available() else 'cpu')
    logger.info(f'使用设备: {device}')
    
    # 加载数据
    logger.info('加载数据...')
    
    # 构建数据路径
    train_data_path = os.path.join(args.data_dir, 'train.json')
    valid_data_path = os.path.join(args.data_dir, 'valid.json')
    test_data_path = os.path.join(args.data_dir, 'test.json')
    
    # 加载数据集
    train_data = MyDataset(args, train_data_path, args.image_dir, mode='train')
    valid_data = MyDataset(args, valid_data_path, args.image_dir, mode='valid')
    test_data = MyDataset(args, test_data_path, args.image_dir, mode='test')
    
    # 初始化模型
    logger.info(f'初始化模型: {args.model_name}')
    
    if args.model_name == 'RCLMFN':
        model = RCLMFN(args)
    else:
        model = MV_CLIP(args)
    
    model.to(device)
    
    # 如果是测试模式
    if args.test_only:
        if not args.test_model_path:
            args.test_model_path = os.path.join(args.save_dir, f'best_model_{args.model_name}.pt')
        
        logger.info(f'加载测试模型: {args.test_model_path}')
        checkpoint = torch.load(args.test_model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # 测试模型
        logger.info('开始测试...')
        test_results = test(args, model, test_data, device)
        return
    
    # 获取优化器和学习率调度器
    optimizer, scheduler = get_optimizer_and_scheduler(args, model, train_data)
    
    # 开始训练
    logger.info('开始训练...')
    best_f1 = train(args, model, device, train_data, valid_data, test_data, optimizer, scheduler)
    
    # 加载最佳模型并测试
    logger.info('加载最佳模型进行测试...')
    best_model_path = os.path.join(args.save_dir, f'best_model_{args.model_name}.pt')
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 测试模型
    logger.info('开始测试...')
    test_results = test(args, model, test_data, device)
    
    logger.info('训练和测试完成!')

if __name__ == "__main__":
    main()