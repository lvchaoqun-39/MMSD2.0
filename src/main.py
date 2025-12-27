import os
import sys
# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import random
import argparse
import numpy as np
import logging
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup, AutoTokenizer

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 移除可能的代理设置
if 'HTTPS_PROXY' in os.environ:
    del os.environ['HTTPS_PROXY']
if 'HTTP_PROXY' in os.environ:
    del os.environ['HTTP_PROXY']
logger.info('已清除代理设置，使用本地网络连接')

# 直接从当前目录导入
from data_set import Dataset
from model import MV_CLIP
from train import train

def set_args():
    parser = argparse.ArgumentParser()
    # 基础参数
    parser.add_argument('--device', default='0', type=str, help='GPU设备索引')
    parser.add_argument('--seed', default=42, type=int, help='随机种子')
    parser.add_argument('--lr', default=5e-5, type=float, help='学习率')
    parser.add_argument('--batch_size', default=16, type=int, help='批大小')
    parser.add_argument('--train_batch_size', default=16, type=int, help='训练批大小（与batch_size保持一致）')
    parser.add_argument('--epoch', default=10, type=int, help='训练轮数')
    parser.add_argument('--num_train_epochs', default=10, type=int, help='训练轮数（与epoch保持一致）')
    parser.add_argument('--early_stop', default=3, type=int, help='早停轮数')
    
    # 模型参数
    parser.add_argument('--layers', default=2, type=int, help='层数')
    parser.add_argument('--text_size', default=512, type=int, help='文本特征维度')
    parser.add_argument('--image_size', default=768, type=int, help='图像特征维度')
    parser.add_argument('--fusion_dim', default=1024, type=int, help='融合特征维度')
    parser.add_argument('--label_number', default=2, type=int, help='类别数量')
    parser.add_argument('--simple_linear', action='store_true', help='是否使用简单线性层')
    parser.add_argument('--dropout_rate', default=0.1, type=float, help='dropout率')
    parser.add_argument('--heads', default=8, type=int, help='注意力头数量')
    
    # 训练策略参数
    parser.add_argument('--use_focal_loss', action='store_true', help='是否使用Focal Loss')
    parser.add_argument('--focal_gamma', default=2.0, type=float, help='Focal Loss的gamma参数')
    parser.add_argument('--use_cosine_scheduler', action='store_true', help='是否使用余弦退火学习率')
    parser.add_argument('--warmup_ratio', default=0.1, type=float, help='预热比例')
    
    # 对比学习参数
    parser.add_argument('--use_contrastive_loss', action='store_true', help='是否使用对比学习损失')
    parser.add_argument('--contrastive_weight', default=0.1, type=float, help='对比损失权重')
    parser.add_argument('--temperature', default=0.07, type=float, help='对比学习温度参数')
    
    # 数据增强参数
    parser.add_argument('--use_data_augmentation', action='store_true', help='是否使用数据增强')
    
    # WandB参数
    parser.add_argument('--use_wandb', default=True, type=bool, help='是否使用WandB')
    parser.add_argument('--project_name', default='MMSD2.0', type=str, help='WandB项目名称')
    
    # 模型名称参数
    parser.add_argument('--tokenizer_name', default='openai/clip-vit-base-patch32', type=str, help='Tokenizer名称')
    
    args = parser.parse_args()
    # 确保train_batch_size与batch_size一致
    if not hasattr(args, 'train_batch_size'):
        args.train_batch_size = args.batch_size
    else:
        # 如果用户指定了不同的值，发出警告并保持一致
        if args.train_batch_size != args.batch_size:
            logger.warning(f'train_batch_size ({args.train_batch_size}) 与 batch_size ({args.batch_size}) 不一致，将使用batch_size的值')
            args.train_batch_size = args.batch_size
    
    # 确保num_train_epochs与epoch一致
    if not hasattr(args, 'num_train_epochs'):
        args.num_train_epochs = args.epoch
    else:
        # 如果用户指定了不同的值，发出警告并保持一致
        if args.num_train_epochs != args.epoch:
            logger.warning(f'num_train_epochs ({args.num_train_epochs}) 与 epoch ({args.epoch}) 不一致，将使用epoch的值')
            args.num_train_epochs = args.epoch
    
    return args

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    args = set_args()
    seed_everything(args.seed)
    
    # 设备设置
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    logger.info(f'使用设备: {device}')
    
    # 加载tokenizer
    logger.info(f'加载tokenizer: {args.tokenizer_name}')
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, local_files_only=False)
        logger.info('Tokenizer加载成功')
    except Exception as e:
        logger.error(f'Tokenizer加载失败: {e}')
        logger.info('尝试使用本地文件加载...')
        # 尝试使用本地路径或替代方案
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, local_files_only=True)
    
    # 数据集加载
    logger.info('加载数据集...')
    train_dataset = Dataset(args, tokenizer, mode='train')
    dev_dataset = Dataset(args, tokenizer, mode='valid')
    test_dataset = Dataset(args, tokenizer, mode='test')
    
    logger.info(f'Train dataset size: {len(train_dataset)}')
    logger.info(f'Dev dataset size: {len(dev_dataset)}')
    logger.info(f'Test dataset size: {len(test_dataset)}')
    
    # 数据加载器
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True, collate_fn=train_dataset.collate_func)
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True, collate_fn=dev_dataset.collate_func)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True, collate_fn=test_dataset.collate_func)
    
    # 模型初始化
    logger.info('初始化模型...')
    model = MV_CLIP(args)
    model = model.to(device)
    
    # 优化器设置
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=1e-8)
    
    # 学习率调度器
    total_steps = len(train_loader) * args.epoch
    warmup_steps = int(total_steps * args.warmup_ratio)
    
    if args.use_cosine_scheduler:
        from transformers import get_cosine_schedule_with_warmup
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    else:
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    
    # 开始训练
    best_f1 = train(args, model, device, train_dataset, dev_dataset, optimizer, scheduler, tokenizer)
    
    logger.info(f'训练完成，最佳F1分数: {best_f1}')

if __name__ == '__main__':
    main()