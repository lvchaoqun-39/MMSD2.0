# 添加猴子补丁以解决torchvision的OrderedDict导入问题
import sys
import collections
if not hasattr(sys.modules['typing'], 'OrderedDict'):
    sys.modules['typing'].OrderedDict = collections.OrderedDict

import os
import json
import torch
import random
import logging
from PIL import Image
from torch.utils.data import Dataset as TorchDataset
from torchvision import transforms
from transformers import AutoTokenizer

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# OrderedDict兼容性导入
try:
    from typing import OrderedDict
except ImportError:
    from collections import OrderedDict

class Dataset(TorchDataset):
    def __init__(self, args, tokenizer, mode='train'):
        self.args = args
        self.tokenizer = tokenizer
        self.mode = mode
        # 安全检查参数是否存在
        self.use_augmentation = getattr(args, 'use_augmentation', False) and mode == 'train'
        
        # 获取项目根目录
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # 尝试多个数据目录
        json_dir_candidates = [
            os.path.join(self.project_root, 'data', 'text_json_final'),
            os.path.join(self.project_root, 'data', 'text_json_clean'),
            os.path.join(self.project_root, 'data')
        ]
        
        self.json_dir = None
        for candidate in json_dir_candidates:
            if os.path.exists(candidate):
                self.json_dir = candidate
                logger.info(f"使用数据目录: {candidate}")
                break
        
        if self.json_dir is None:
            raise FileNotFoundError("未找到有效的数据目录")
        
        # 加载数据
        self.data = self._load_data()
        
        # 图像变换
        self.transform = self._get_transforms()
        
        logger.info(f"{mode} dataset loaded with {len(self.data)} samples")
    
    def _load_data(self):
        # 加载对应模式的数据，支持更多的模式名称
        data_files = {
            'train': 'train.json',
            'dev': 'valid.json',
            'valid': 'valid.json',  # 直接支持valid作为模式名称
            'test': 'test.json'
        }
        
        if self.mode not in data_files:
            raise ValueError(f"Invalid mode: {self.mode}")
        
        file_path = os.path.join(self.json_dir, data_files[self.mode])
        
        # 如果在当前目录找不到，尝试直接使用文件名（兼容旧路径）
        if not os.path.exists(file_path):
            file_path = os.path.join(os.path.dirname(self.json_dir), data_files[self.mode])
        
        # 再尝试其他可能的路径
        if not os.path.exists(file_path):
            file_path = os.path.join(os.path.dirname(os.path.dirname(self.json_dir)), 'data', data_files[self.mode])
        
        # 最后尝试直接使用相对路径
        if not os.path.exists(file_path):
            file_path = os.path.join('..', 'data', data_files[self.mode])
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"数据文件不存在: {file_path}")
        
        logger.info(f"加载数据文件: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 不再过滤样本，直接返回所有数据
            logger.info(f"加载的数据样本数: {len(data)}")
            return data
        except Exception as e:
            logger.error(f"加载数据文件时出错: {str(e)}")
            raise
    
    def _get_transforms(self):
        transform_list = []
        
        # 基础变换
        transform_list.append(transforms.Resize((224, 224)))
        
        # 数据增强（仅训练集）
        if self.use_augmentation:
            transform_list.extend([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.RandomRotation(10),
                transforms.RandomCrop(size=224, padding=10)
            ])
        
        # 必须的变换
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        return transforms.Compose(transform_list)
    
    def _find_image_path(self, image_name):
        # 尝试所有可能的图像路径
        possible_paths = [
            # 项目根目录下的pictures文件夹
            os.path.join(self.project_root, 'pictures', image_name),
            # data目录下的dataset_image文件夹
            os.path.join(self.project_root, 'data', 'dataset_image', image_name),
            # 当前目录下的pictures文件夹
            os.path.join('..', 'pictures', image_name),
            # 当前目录下的dataset_image文件夹
            os.path.join('..', 'dataset_image', image_name)
        ]
        
        # 遍历所有可能的路径
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        # 如果找不到，返回第一个可能的路径（让上层处理错误）
        return possible_paths[0]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        item = self.data[index]
        
        # 处理文本
        text = item['text']
        
        # 简单的文本增强（避免使用nltk）
        if self.use_augmentation and random.random() < 0.3:
            # 随机插入空格
            words = text.split()
            if len(words) > 1:
                # 随机重复一个词
                pos = random.randint(0, len(words) - 1)
                words.insert(pos, words[pos])
                text = ' '.join(words)
        
        # 文本编码
        text_encoding = self.tokenizer(
            text,
            max_length=77,  # CLIP模型的最大长度
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        # 处理图像
        image_name = item.get('image', '')
        if not image_name:
            # 如果没有图像名，使用默认的空白图像
            image = torch.zeros(3, 224, 224)
        else:
            image_path = self._find_image_path(image_name)
            
            try:
                image = Image.open(image_path).convert('RGB')
                image = self.transform(image)
            except Exception as e:
                logger.error(f"处理图像 {image_path} 时出错: {str(e)}")
                # 创建空白图像作为替代
                image = torch.zeros(3, 224, 224)
        
        # 处理标签
        label = int(item.get('label', 0))
        
        return {
            'input_ids': text_encoding['input_ids'].squeeze(0),
            'attention_mask': text_encoding['attention_mask'].squeeze(0),
            'pixel_values': image,
            'labels': torch.tensor(label, dtype=torch.long)
        }
    
    def collate_func(self, batch):
        # 确保batch中的所有张量都有相同的维度
        input_ids = torch.stack([item['input_ids'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])
        pixel_values = torch.stack([item['pixel_values'] for item in batch])
        labels = torch.stack([item['labels'] for item in batch])
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'pixel_values': pixel_values,
            'labels': labels
        }