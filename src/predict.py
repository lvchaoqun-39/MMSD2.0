import os
from model import CLIPClassificationModel_transformer
from transformers import CLIPProcessor
from torch.utils.data import DataLoader
import torch
from data_set import MyDataset
import argparse
from tqdm import tqdm
import json
import numpy as np
from sklearn import metrics


def predict(args, model, device, data, processor, pre = None):

    data_loader = DataLoader(data, batch_size=args.test_batch_size, collate_fn=MyDataset.collate_func,shuffle=False) # 把数据集 data 包装成可迭代的批数据加载器
    n_correct, n_total = 0, 0 # 计数已预测正确的样本数量和总样本数量
    t_targets_all, t_outputs_all = None, None # 用于后续汇总所有 batch 的真实标签与模型输出

    model.eval() # 切换到评估模式；关闭 Dropout 的随机性； 让 BatchNorm 使用运行时统计量而不是用当前 batch 统计
    data = []
    image = []
    text = []
    logit = []
    with open(pre,'w',encoding='utf-8') as fout:
        with torch.no_grad(): # 关闭梯度计算，减少显存占用、加快推理；用于测试/预测阶段
            for i_batch, t_batch in enumerate(data_loader):
                text_list, image_list, label_list, id_list = t_batch
                image.extend(id_list)
                text.extend(text_list) # 把当前 batch 的样本 id、文本内容累积到外部列表里，方便最终对齐保存
                inputs = processor(text=text_list, images=image_list, padding='max_length', truncation=True, max_length=args.max_len, return_tensors="pt").to(device) # 把“原始文本列表 + 图像列表”转换为模型可用的张量输入
                labels = torch.tensor(label_list).to(device) # 把标签列表转成张量并搬到设备

                t_targets = labels # 本 batch 的真实标签
                loss, t_outputs = model(inputs,labels=labels) # ：调用模型前向， t_outputs ：模型输出 logits
                logit.extend(t_outputs.cpu().tolist()) # 把当前 batch 的 logits 转到 CPU 并转成 Python list，累积保存，便于后续写文件/分析

                n_correct += (torch.argmax(t_outputs, -1) == t_targets).sum().item() # 计算并累计 batch 的预测正确数与总样本数
                n_total += len(t_outputs) # 累计样本总数

                # 把所有 batch 的标签和输出拼接起来，形成“全量”的 t_targets_all 、 t_outputs_all
                if t_targets_all is None:
                    t_targets_all = t_targets
                    t_outputs_all = t_outputs
                else:
                    t_targets_all = torch.cat((t_targets_all, t_targets), dim=0)
                    t_outputs_all = torch.cat((t_outputs_all, t_outputs), dim=0)
        if pre != None:
            predict = torch.argmax(t_outputs_all, -1).cpu().numpy().tolist() # 对每条样本的 logits 取最大值所在类别作为预测标签
            label = t_targets_all.cpu().numpy().tolist() # 真实标签列表
            for image_, text_, label_, predict_, logi_ in zip(image, text,label, predict, logit):
                data.append({'image_id':image_, 'text':text_, 'label':label_, 'predict':predict_, 'logit':logi_}) 
        json.dump({'data': data}, fout)            
        
        acc = n_correct / n_total
        f1 = metrics.f1_score(t_targets_all.cpu(), torch.argmax(t_outputs_all, -1).cpu())
        precision =  metrics.precision_score(t_targets_all.cpu(),torch.argmax(t_outputs_all, -1).cpu())
        recall = metrics.recall_score(t_targets_all.cpu(),torch.argmax(t_outputs_all, -1).cpu())
        f1_ = metrics.f1_score(t_targets_all.cpu(), torch.argmax(t_outputs_all, -1).cpu(), labels=[0, 1],average='macro')
        precision_ =  metrics.precision_score(t_targets_all.cpu(),torch.argmax(t_outputs_all, -1).cpu(), labels=[0, 1],average='macro')
        recall_ = metrics.recall_score(t_targets_all.cpu(),torch.argmax(t_outputs_all, -1).cpu(), labels=[0, 1],average='macro')
        print("test_acc is {}, macro_test_f1 is {}, macro_test_precision is {}, macro_test_recall is {}, micro_test_f1 is {}, micro_test_precision is {}, micro_test_recall is {}".format(acc, f1_, precision_, recall_, f1, precision, recall))



def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='0', type=str, help='device number')
    parser.add_argument('--max_len', type=int, default=77, help='max length of text')
    parser.add_argument('--text_size', default=512, type=int, help='text hidden size')
    parser.add_argument('--image_size', default=768, type=int, help='image hidden size')
    parser.add_argument('--dropout_rate', default=0.5, type=float, help='dropout rate')
    parser.add_argument('--label_number', type=int, default=2, help='number of classification labels')
    parser.add_argument('--test_batch_size', type=int, default=8, help='batch size for text phase')
    parser.add_argument('--model_path', type=str, default="../output_dir/MV_CLIP", help='save model dpath')
    parser.add_argument('--save_file', type=str, default="result.json", help='save result path')
    parser.add_argument('--text_name', default='text_json_final', type=str, help='the text data folder name')
    parser.add_argument('--layers', default=3, type=int, help='number of layers of transformers')
    parser.add_argument('--simple_linear', default=False, type=bool, help='linear implementation choice')
    return parser.parse_args()


def main():
    args = set_args() # 解析/构造运行参数
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    device = torch.device("cuda" if torch.cuda.is_available() and int(args.device) >= 0 else "cpu")

    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32") # 加载 CLIP 的处理器（负责把文本 tokenize、把图片做 resize/normalize，并打包成张量输入）
    model = CLIPClassificationModel_transformer(args) # 构建自定义的分类模型（基于 CLIP + transformer/分类头，具体在model.py里）

    test_data = MyDataset(mode='test', text_name=args.text_name, limit=None) # 构建测试集数据集对象

    model.load_state_dict(torch.load(os.path.join(args.model_path, "model.pt"), map_location="cpu")) # 从 args.model_path/model.pt 读取权重；先映射到 CPU，避免因设备不一致加载失败
    model.to(device)
    model.eval()

    predict(args, model, device, test_data, processor, pre=args.save_file)


if __name__ == '__main__':
    main()
