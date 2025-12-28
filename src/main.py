import os

os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7897'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7897'
import sys
import random
import numpy as np
import torch
from datetime import datetime
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AdamW, Adafactor, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from data_set import Dataset
from model import MV_CLIP
import logging
import argparse

# 尝试导入wandb，如果不可用则设置为None
try:
    import wandb
except ImportError:
    wandb = None
    print("警告: wandb模块未找到，将禁用wandb功能")

# 设置环境变量以完全禁用wandb实时同步功能
# 配置wandb环境变量 - 允许实时同步
os.environ['WANDB_MODE'] = 'online'
os.environ['WANDB_SILENT'] = 'true'
os.environ['WANDB_CONSOLE'] = 'off'
os.environ['WANDB_TIMEOUT'] = '600'
os.environ['WANDB_HTTP_TIMEOUT'] = '300'
os.environ['WANDB_AUTO_SYNC'] = 'true'
os.environ['WANDB_DISABLE_GIT'] = 'true'

# 不禁用代理设置，以允许网络连接

# 设置随机种子
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def set_args():
    parser = argparse.ArgumentParser()
    # 设备设置
    parser.add_argument('--device', default='0', type=str, help='使用的GPU编号')
    # 数据设置
    parser.add_argument('--data_path', default='data/MMSD/process_data', type=str, help='数据路径')
    parser.add_argument('--max_seq_length', default=128, type=int, help='文本最大长度')
    parser.add_argument('--batch_size', default=32, type=int, help='批大小')
    # 模型设置
    parser.add_argument('--model', default='MV_CLIP', type=str, help='模型名称')
    parser.add_argument('--visual_model', default='resnet50', type=str, help='视觉模型')
    parser.add_argument('--audio_model', default='VGGish', type=str, help='音频模型')
    parser.add_argument('--text_model', default='bert-base-uncased', type=str, help='文本模型')
    parser.add_argument('--tokenizer_name', default='bert-base-uncased', type=str, help='tokenizer名称')
    # 模型内部参数 - 新增
    parser.add_argument('--text_size', default=512, type=int, help='文本特征维度，CLIP模型实际为512')
    parser.add_argument('--image_size', default=512, type=int, help='图像特征维度，CLIP模型实际为512')
    parser.add_argument('--heads', default=8, type=int, help='注意力头数量')
    parser.add_argument('--fusion_dim', default=1024, type=int, help='融合层维度')
    parser.add_argument('--dropout_rate', default=0.1, type=float, help='dropout率')
    parser.add_argument('--label_number', default=2, type=int, help='标签数量，二分类问题设为2')
    # 损失函数参数 - 新增
    parser.add_argument('--use_contrastive_loss', default=False, type=bool, help='是否使用对比损失')
    parser.add_argument('--temperature', default=0.07, type=float, help='对比损失温度参数')
    parser.add_argument('--contrastive_weight', default=0.1, type=float, help='对比损失权重')
    parser.add_argument('--use_focal_loss', default=False, type=bool, help='是否使用Focal Loss')
    parser.add_argument('--focal_gamma', default=2.0, type=float, help='Focal Loss的gamma参数')
    # 训练设置
    parser.add_argument('--seed', default=42, type=int, help='随机种子')
    parser.add_argument('--num_train_epochs', default=10, type=int, help='训练轮数')
    parser.add_argument('--lr', default=1e-5, type=float, help='学习率')
    parser.add_argument('--weight_decay', default=0.01, type=float, help='权重衰减')
    parser.add_argument('--warmup_ratio', default=0.1, type=float, help='预热比例')
    parser.add_argument('--early_stop', default=3, type=int, help='早停轮数')
    # Wandb参数
    parser.add_argument('--use_wandb', default=True, type=bool, help='是否使用wandb')
    parser.add_argument('--project_name', default='MMSD2.0', type=str, help='wandb项目名称')
    parser.add_argument('--logging_steps', default=5, type=int, help='日志记录步数')
    # 输出设置
    parser.add_argument('--output_dir', default='outputs', type=str, help='输出目录')
    parser.add_argument('--save_model', default=True, type=bool, help='是否保存模型')
    args = parser.parse_args()
    return args

# 设置日志
def setup_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

logger = setup_logger()

def main():
    args = set_args()
    seed_everything(args.seed)
    
    # 如果wandb模块不可用，自动禁用wandb功能
    if wandb is None:
        logger.warning("wandb模块不可用，已自动禁用wandb功能")
        args.use_wandb = False
    
    # 添加wandb初始化代码
    if args.use_wandb and wandb is not None:
        try:
            # 允许实时网络连接
            os.environ['WANDB_MODE'] = 'online'
            
            wandb.init(
                project=args.project_name,
                config=vars(args),  # 将所有参数记录到wandb
                name=f"run_{args.seed}_{args.model}",  # 为运行命名
                reinit=True,  # 允许重新初始化
                mode='online'  # 实时模式
            )
            logger.info(f'Wandb已在在线模式下初始化，项目名称: {args.project_name}')
            logger.info(f'数据将实时同步到wandb服务器')
            
            # 记录一些关键超参数
            wandb.config.update({
                "learning_rate": args.lr,
                "batch_size": args.batch_size,
                "epochs": args.num_train_epochs,
                "model": args.model
            })
        except Exception as e:
            logger.error(f'Wandb初始化失败: {e}')
            logger.warning('已完全禁用wandb功能以避免网络请求')
            args.use_wandb = False  # 完全禁用wandb
    
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
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    
    # 根据不同的模型选择不同的优化器
    if 'clip' in args.text_model.lower():
        # CLIP模型使用Adafactor优化器
        optimizer = Adafactor(
            optimizer_grouped_parameters,
            lr=args.lr,
            eps=(1e-30, 1e-3),
            clip_threshold=1.0,
            decay_rate=-0.8,
            beta1=None,
            weight_decay=0.0,
            relative_step=False,
            scale_parameter=False,
            warmup_init=False
        )
    else:
        # 其他模型使用AdamW优化器
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=args.lr,
            eps=1e-8
        )
    
    # 学习率调度器
    total_steps = len(train_loader) * args.num_train_epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    
    # 创建输出目录
    if args.output_dir and not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # 导入训练和评估函数
    from train import train, evaluate_acc_f1
    
    # 开始训练
    logger.info("开始训练...")
    
    # 调用正确的train函数，传递optimizer和scheduler参数
    train(args, model, device, train_dataset, dev_dataset, optimizer, scheduler, tokenizer)
    
    # 测试阶段
    logger.info("开始测试...")
    
    # 加载保存的最佳模型
    path_to_save = os.path.join(args.output_dir, args.model)
    model.load_state_dict(torch.load(os.path.join(path_to_save, 'model.pt'), map_location=device))
    
    # 调用evaluate_acc_f1函数进行测试，确保参数匹配
    test_acc, test_f1, test_precision, test_recall = evaluate_acc_f1(
        args, model, device, test_dataset, tokenizer, mode='test'
    )
    
    logger.info(f"Test Accuracy: {test_acc}, Test F1: {test_f1}")
    
    # 关闭wandb
    if args.use_wandb and wandb is not None:
        try:
            wandb.finish()
        except Exception as e:
            logger.error(f"Error closing wandb: {e}")

    return 0

if __name__ == '__main__':
    main()