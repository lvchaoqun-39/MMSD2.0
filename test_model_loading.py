import os
import torch
from model import MV_CLIP
import argparse  # 直接导入argparse
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def set_args():
    parser = argparse.ArgumentParser()
    # 设备设置
    parser.add_argument('--device', default='0', type=str, help='使用的GPU编号')
    # 输出设置
    parser.add_argument('--output_dir', default='outputs', type=str, help='输出目录')
    # 模型设置（只保留必要的参数）
    parser.add_argument('--model', default='MV_CLIP', type=str, help='模型名称')
    parser.add_argument('--text_size', default=512, type=int, help='文本特征维度')
    parser.add_argument('--image_size', default=512, type=int, help='图像特征维度')
    parser.add_argument('--heads', default=8, type=int, help='注意力头数量')
    parser.add_argument('--fusion_dim', default=1024, type=int, help='融合层维度')
    parser.add_argument('--dropout_rate', default=0.1, type=float, help='dropout率')
    parser.add_argument('--label_number', default=2, type=int, help='标签数量')
    args = parser.parse_args()
    return args

def test_model_loading():
    # 获取命令行参数
    args = set_args()
    
    # 设备设置
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    logger.info(f'使用设备: {device}')
    
    # 初始化模型
    logger.info('初始化模型...')
    model = MV_CLIP(args)
    model = model.to(device)
    
    # 测试我们修改的模型加载代码
    logger.info("测试模型加载...")
    
    # 查找outputs目录下最新的best_model_epoch_X.pth文件
    best_model_files = [f for f in os.listdir(args.output_dir) if f.startswith('best_model_epoch_') and f.endswith('.pth')]
    if not best_model_files:
        logger.error("找不到最佳模型文件")
        return 1
    
    # 按epoch号排序，取最新的模型
    best_model_files.sort(key=lambda x: int(x.split('_')[3].split('.')[0]), reverse=True)
    best_model_path = os.path.join(args.output_dir, best_model_files[0])
    logger.info(f"加载最佳模型: {best_model_path}")
    
    try:
        # 尝试加载模型
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        logger.info("模型加载成功！")
        return 0
    except Exception as e:
        logger.error(f"模型加载失败: {e}")
        return 1

if __name__ == '__main__':
    test_model_loading()

