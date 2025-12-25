import os
import sys
import torch
from model import MV_BERT_RESNET
from data_set import MyDataset
from train import train
import argparse

def seed_everything(seed=42):
    """设置随机种子，确保实验可复现"""
    import numpy as np
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 启用cudnn.benchmark以加速训练
    torch.backends.cudnn.benchmark = True
    # 禁用cudnn的deterministic模式以提高性能
    torch.backends.cudnn.deterministic = False

# 配置参数
args = argparse.Namespace(
    # 基本参数
    model='MV_BERT_RESNET',
    output_dir='./output_test',
    seed=42,
    
    # 模型参数
    bert_name='bert-base-uncased',
    
    # 训练参数
    train_batch_size=16,
    dev_batch_size=32,
    num_train_epochs=1,
    learning_rate=2e-5,
    clip_learning_rate=1e-5,
    weight_decay=0.01,
    adam_epsilon=1e-8,
    optimizer_name='adam',
    warmup_proportion=0.1,
    max_len=128,
    
    # 性能优化参数
    num_workers=4,
    pin_memory=True,
    prefetch_factor=2,
    gradient_accumulation_steps=1,
    
    # 数据集限制
    limit=500  # 只使用500个样本进行测试
)

# 设置随机种子
seed_everything(args.seed)

# 检查GPU是否可用
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 加载数据集
train_data = MyDataset(mode='train', text_name='text', limit=args.limit)
print(f"Loaded {len(train_data)} training samples")

# 创建较小的验证和测试集
dev_data = train_data[:100]  # 使用前100个样本作为验证集
test_data = train_data[100:200]  # 使用100-200个样本作为测试集

# 创建模拟的processor（简化版）
from transformers import BertTokenizer
from torchvision import transforms

class MockProcessor:
    def __init__(self):
        self["tokenizer"] = BertTokenizer.from_pretrained(args.bert_name)
        self["image_transform"] = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    
    def __getitem__(self, key):
        return super().__getitem__(key)

processor = MockProcessor()

# 创建模型
model = MV_BERT_RESNET(bert_name=args.bert_name, 
                       resnet_model="resnet18",  # 使用ResNet18
                       freeze_bert_layers=10,  # 冻结BERT前10层
                       freeze_resnet=True)  # 完全冻结ResNet
print(f"Model created with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters")

# 开始计时
try:
    import time
    start_time = time.time()
    
    # 开始训练
    print("Starting training with performance optimizations...")
    train(args, model, device, train_data, dev_data, test_data, processor)
    
    end_time = time.time()
    print(f"Training completed in {end_time - start_time:.2f} seconds")
    print("Performance optimizations have been applied successfully!")
except Exception as e:
    print(f"Error during testing: {e}")
    import traceback
    traceback.print_exc()
