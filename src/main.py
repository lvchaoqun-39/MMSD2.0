import os

os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7897'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7897'
# os.environ["CUDA_VISIBLE_DEVICES"] = '2'



from model import MV_CLIP
from train import train
from data_set import MyDataset
import torch
import argparse
import random
import numpy as np
from transformers import CLIPProcessor
import wandb
import pickle
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='1', type=str, help='device number')
    parser.add_argument('--model', default='MV_CLIP', type=str, help='the model name', choices=['MV_CLIP']) # 使用的模型
    parser.add_argument('--text_name', default='text_json_final', type=str, help='the text data folder name') # 文本数据文件夹名
    parser.add_argument('--simple_linear', default=False, type=bool, help='linear implementation choice')
    parser.add_argument('--num_train_epochs', default=10, type=int, help='number of train epoched') # 训练轮数
    parser.add_argument('--train_batch_size', default=32, type=int, help='batch size in train phase') # 训练时每次批量处理样本数量
    parser.add_argument('--dev_batch_size', default=32, type=int, help='batch size in dev phase') # 验证时每次批量处理样本数量
    parser.add_argument('--label_number', default=2, type=int, help='the number of classification labels') # 分类标签数量
    parser.add_argument('--text_size', default=512, type=int, help='text hidden size') # 文本隐藏层大小
    parser.add_argument('--image_size', default=768, type=int, help='image hidden size') # 图像隐藏层大小
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.") # Adam优化器的epsilon参数
    parser.add_argument("--optimizer_name", type=str, default='adam',
                        help="use which optimizer to train the model.") # 使用的优化器
    parser.add_argument('--learning_rate', default=5e-4, type=float, help='learning rate for modules expect CLIP') # 学习率
    parser.add_argument('--clip_learning_rate', default=1e-6, type=float, help='learning rate for CLIP') # CLIP的学习率
    parser.add_argument('--max_len', default=77, type=int, help='max len of text based on CLIP') # 模型能处理的输入文本序列的最大长度
    parser.add_argument('--layers', default=3, type=int, help='number of transform layers') # transform层数量
    parser.add_argument('--max_grad_norm', default=5.0, type=float, help='grad clip norm') # 梯度裁剪范数（梯度裁剪阈值，用于防止梯度爆炸）
    parser.add_argument('--weight_decay', default=0.05, type=float, help='weight decay') # 权重衰减（防止过拟合）
    parser.add_argument('--warmup_proportion', default=0.2, type=float, help='warm up proportion') # 预热比例
    parser.add_argument('--dropout_rate', default=0.1, type=float, help='dropout rate') # 丢弃率（随机丢弃神经元）
    parser.add_argument('--output_dir', default='../output_dir/', type=str, help='the output path') # 输出路径
    parser.add_argument('--limit', default=None, type=int, help='the limited number of training examples') # 训练样本数量限制
    parser.add_argument('--seed', type=int, default=42, help='random seed') # 随机种子
    return parser.parse_args()


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def main():
    args = set_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    device = torch.device("cuda" if torch.cuda.is_available() and int(args.device) >= 0 else "cpu")

    seed_everything(args.seed)

    wandb.init(
        project="MMSD2.0",
        notes="mm",
        tags=["mm"],
        config=vars(args),
    )
    wandb.watch_called = False  

    train_data = MyDataset(mode='train', text_name=args.text_name, limit=None)
    dev_data = MyDataset(mode='valid', text_name=args.text_name, limit=None)
    test_data = MyDataset(mode='test', text_name=args.text_name, limit=None)

    if args.model == 'MV_CLIP':
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        model = MV_CLIP(args)
    else:
        raise RuntimeError('Error model name!')

    model.to(device)
    wandb.watch(model, log="all")

    train(args, model, device, train_data, dev_data, test_data, processor)



if __name__ == '__main__':
    main()
