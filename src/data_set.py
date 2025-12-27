import os
import json
import sys
import collections  # 确保collections模块被导入
# Python 3.7.1兼容的OrderedDict导入
if sys.version_info < (3, 8):
    # 对于Python 3.7.1，OrderedDict在collections中
    from collections import OrderedDict
    # 为了兼容性，在typing模块中添加OrderedDict引用
    import typing
    if not hasattr(typing, 'OrderedDict'):
        typing.OrderedDict = OrderedDict
else:
    # 对于Python 3.8及以上，OrderedDict在typing中
    from typing import OrderedDict

from PIL import Image
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer, CLIPProcessor
import torchvision.transforms as transforms

class MyDataset(Dataset):
    def __init__(self, args, data_dir, image_dir, mode='train'):
        self.args = args
        self.data_dir = data_dir
        self.image_dir = image_dir
        self.mode = mode
        
        # 加载数据
        self.data = []
        with open(data_dir, 'r', encoding='utf-8') as f:
            # 一次性加载整个JSON数组
            self.data = json.load(f)

        # 检查图像文件是否存在
        self.filtered_data = []
        for item in self.data:
            img_path = os.path.join(image_dir, f"{item['image_id']}.jpg")
            if os.path.exists(img_path):
                self.filtered_data.append(item)
        
        # 数据统计
        print(f"{mode} data loaded: {len(self.filtered_data)} samples")
        
        # 初始化处理器
        if hasattr(args, 'model_name') and args.model_name == 'RCLMFN':
            # 对于RCLMFN模型，使用BERT Tokenizer和CLIP图像处理
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', local_files_only=True)
            self.image_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", local_files_only=True)
        else:
            # 对于MV_CLIP模型，使用CLIP处理器
            self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", local_files_only=True)
        
        # 图像变换（针对非CLIP处理器情况）
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def __getitem__(self, idx):
        item = self.filtered_data[idx]
        text = item['text']
        image_id = item['image_id']
        label = int(item['label'])
        
        # 加载图像，添加错误处理
        img_path = os.path.join(self.image_dir, f"{image_id}.jpg")
        try:
            image = Image.open(img_path).convert('RGB')
            # 验证图像是否完整
            image.load()
        except Exception as e:
            print(f"警告: 无法加载图像 {img_path}, 错误: {str(e)}. 使用空白图像代替。")
            # 创建空白图像作为替代
            image = Image.new('RGB', (224, 224), color='white')
        
        # 根据模型类型进行不同的处理
        if hasattr(self.args, 'model_name') and self.args.model_name == 'RCLMFN':
            # RCLMFN模型的数据处理 - 返回嵌套字典格式
            text_inputs = self.tokenizer(
                text,
                max_length=self.args.max_seq_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            # 处理图像
            image_inputs = self.image_processor(
                images=image,
                return_tensors="pt"
            )
            
            # 准备嵌套字典格式的输入
            inputs = {
                'text': {
                    'input_ids': text_inputs['input_ids'].squeeze(0),
                    'attention_mask': text_inputs['attention_mask'].squeeze(0)
                },
                'image': {
                    'pixel_values': image_inputs['pixel_values'].squeeze(0)
                }
            }
        else:
            # MV_CLIP模型的数据处理
            inputs = self.processor(
                text=text,
                images=image,
                return_tensors="pt",
                padding=True,
                truncation=True
            )
            
            # 移除batch维度
            for k, v in inputs.items():
                inputs[k] = v.squeeze(0)
        
        return inputs, torch.tensor(label, dtype=torch.long)
    
    def __len__(self):
        return len(self.filtered_data)
    
    @staticmethod
    def collate_func(batch):
        # 组装批次数据
        inputs_list, labels_list = zip(*batch)
        
        # 检查是否为嵌套字典格式（RCLMFN模型）
        if isinstance(inputs_list[0], dict) and 'text' in inputs_list[0] and 'image' in inputs_list[0]:
            # 处理嵌套字典格式
            collated_inputs = {
                'text': {},
                'image': {}
            }
            
            # 处理文本部分
            for key in inputs_list[0]['text'].keys():
                collated_inputs['text'][key] = torch.stack([inputs['text'][key] for inputs in inputs_list])
            
            # 处理图像部分
            for key in inputs_list[0]['image'].keys():
                collated_inputs['image'][key] = torch.stack([inputs['image'][key] for inputs in inputs_list])
        else:
            # 处理普通字典格式（MV_CLIP模型）
            collated_inputs = {}
            for key in inputs_list[0].keys():
                collated_inputs[key] = torch.stack([inputs[key] for inputs in inputs_list])
        
        # 合并标签
        collated_labels = torch.stack(labels_list)
        
        return collated_inputs, collated_labels