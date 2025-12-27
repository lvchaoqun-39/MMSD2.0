import os
import time
import json
import torch
from torch.utils.data import DataLoader
import logging
from tqdm import tqdm, trange
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
import wandb
import numpy as np

# 导入数据集类
from data_set import MyDataset

# 配置日志
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def train(args, model, device, train_data, dev_data, test_data, optimizer, scheduler):
    """
    训练模型
    """
    # 创建保存目录
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    # 初始化wandb
    if args.use_wandb:
        run = wandb.init(
            project="MMSD2.0",
            name=f"{args.model_name}_{args.dataset}_{time.strftime('%Y%m%d_%H%M%S')}",
            config=vars(args)
        )
    
    # 准备数据加载器
    train_dataloader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=getattr(train_data, 'collate_func', None)
    )
    
    dev_dataloader = DataLoader(
        dev_data,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=getattr(dev_data, 'collate_func', None)
    )
    
    # 移动模型到设备
    model.to(device)
    
    # 获取processor
    processor = getattr(train_data, 'processor', None)
    
    # 训练参数
    best_dev_acc = 0
    best_epoch = 0
    
    # 训练循环
    for epoch in range(args.num_epochs):
        print(f"Epoch {epoch + 1}/{args.num_epochs}")
        
        # 训练阶段
        model.train()
        total_loss = 0
        start_time = time.time()
        
        for step, batch in enumerate(tqdm(train_dataloader, desc="Training")):
            # 准备输入 - 根据data_set.py返回的格式调整
            inputs, labels = batch
            
            # 将数据移动到设备
            if isinstance(inputs, dict) and 'text' in inputs and 'image' in inputs:
                # RCLMFN模型的嵌套字典格式
                for key in inputs['text']:
                    inputs['text'][key] = inputs['text'][key].to(device)
                for key in inputs['image']:
                    inputs['image'][key] = inputs['image'][key].to(device)
            else:
                # MV_CLIP模型的普通字典格式
                for key in inputs:
                    inputs[key] = inputs[key].to(device)
            labels = labels.to(device)
            
            # 梯度清零
            optimizer.zero_grad()
            
            # 前向传播
            # if args.model_name == 'RCLMFN':
            #     outputs = model(**inputs)
            # else:
            #     # MV_CLIP模型
            #     outputs = model(**inputs)
            outputs = model(inputs, labels)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs
            loss = outputs[0] if isinstance(outputs, tuple) else outputs
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            
            # 更新学习率
            if scheduler is not None and args.optimizer == 'AdamW':
                scheduler.step()
            elif scheduler is not None and args.optimizer == 'Adafactor':
                scheduler.step()
            
            total_loss += loss.item()
            
            # 记录wandb
            if args.use_wandb and (step + 1) % args.log_step == 0:
                wandb.log({
                    "train_loss": loss.item(),
                    "learning_rate": optimizer.param_groups[0]['lr'],
                    "step": step + epoch * len(train_dataloader)
                })
        
        # 计算平均损失
        avg_train_loss = total_loss / len(train_dataloader)
        train_time = time.time() - start_time
        
        print(f"Train Loss: {avg_train_loss:.4f}, Time: {train_time:.2f}s")
        
        # 验证阶段
        dev_acc, dev_f1_macro, dev_precision_macro, dev_recall_macro = evaluate_acc_f1(
            args, model, device, dev_data, processor, macro=True, mode='dev'
        )
        
        print(f"Dev Acc: {dev_acc:.4f}, F1 Macro: {dev_f1_macro:.4f}")
        
        # 记录wandb
        if args.use_wandb:
            wandb.log({
                "epoch": epoch + 1,
                "avg_train_loss": avg_train_loss,
                "dev_acc": dev_acc,
                "dev_f1_macro": dev_f1_macro,
                "dev_precision_macro": dev_precision_macro,
                "dev_recall_macro": dev_recall_macro
            })
        
        # 保存最佳模型
        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            best_epoch = epoch + 1
            
            # 保存模型
            path_to_save = os.path.join(args.save_dir, args.model_name)
            if not os.path.exists(path_to_save):
                os.makedirs(path_to_save)
            
            model_to_save = (model.module if hasattr(model, "module") else model)
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model_to_save.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_dev_acc
            }, os.path.join(path_to_save, 'best_model.pt'))
            
            print(f"最佳模型已保存，准确率: {best_dev_acc:.4f}，轮次: {best_epoch}")
            
            # 在测试集上评估
            test_acc, test_f1_macro, test_precision_macro, test_recall_macro = evaluate_acc_f1(
                args, model, device, test_data, processor, macro=True, mode='test'
            )
            test_acc_micro, test_f1_micro, test_precision_micro, test_recall_micro = evaluate_acc_f1(
                args, model, device, test_data, processor, macro=False, mode='test'
            )
            
            print(f"Test Acc: {test_acc:.4f}, F1 Macro: {test_f1_macro:.4f}, F1 Micro: {test_f1_micro:.4f}")
            
            if args.use_wandb:
                wandb.log({
                    'test_acc': test_acc,
                    'macro_test_f1': test_f1_macro,
                    'macro_test_precision': test_precision_macro,
                    'macro_test_recall': test_recall_macro,
                    'micro_test_f1': test_f1_micro,
                    'micro_test_precision': test_precision_micro,
                    'micro_test_recall': test_recall_micro
                })
        
        # 保存每个epoch的模型
        if (epoch + 1) % args.save_epoch == 0:
            path_to_save = os.path.join(args.save_dir, args.model_name)
            if not os.path.exists(path_to_save):
                os.makedirs(path_to_save)
            
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, os.path.join(path_to_save, f'model_epoch_{epoch + 1}.pt'))
        
        torch.cuda.empty_cache()
    
    # 记录最佳结果
    if args.use_wandb:
        wandb.log({"best_dev_acc": best_dev_acc, "best_epoch": best_epoch})
        run.finish()
    
    print(f"训练完成。最佳准确率: {best_dev_acc:.4f}，轮次: {best_epoch}")
    
    return best_dev_acc


def evaluate_acc_f1(args, model, device, data, processor, macro=False, pre=None, mode='test'):
    """
    评估模型性能
    """
    data_loader = DataLoader(
        data, 
        batch_size=args.batch_size, 
        collate_fn=MyDataset.collate_func, 
        shuffle=False
    )
    
    model.eval()
    n_correct, n_total = 0, 0
    t_targets_all, t_outputs_all = None, None
    sum_loss = 0.
    sum_step = 0
    
    with torch.no_grad():
        for i_batch, t_batch in enumerate(tqdm(data_loader, desc="Evaluating")):
            # 修改数据解包方式，适应新的数据格式
            inputs, labels = t_batch
            
            # 数据预处理
            # 根据模型类型进行不同的处理
            if args.model_name == 'MV_CLIP':
                # 对于MV_CLIP模型，inputs已经是处理好的格式
                # 直接移动到设备上
                model_inputs = {k: v.to(device) for k, v in inputs.items()}
            else:  # RCLMFN
                # 对于RCLMFN模型，inputs是嵌套字典格式
                model_inputs = {
                    'text': {k: v.to(device) for k, v in inputs['text'].items()},
                    'image': {k: v.to(device) for k, v in inputs['image'].items()}
                }
            
            # 模型推理
            outputs = model(model_inputs, labels.to(device))
            loss = outputs[1] if isinstance(outputs, tuple) else outputs
            logits = outputs[0] if isinstance(outputs, tuple) else outputs
            
            # 计算预测结果
            _, predicted = torch.max(logits, 1)
            n_correct += (predicted == labels.to(device)).sum().item()
            n_total += labels.size(0)
            
            # 累加损失
            sum_loss += loss.item()
            sum_step += 1
            
            # 收集结果用于F1计算
            if t_targets_all is None:
                t_targets_all = labels.cpu()
                t_outputs_all = logits.cpu()
            else:
                t_targets_all = torch.cat((t_targets_all, labels.cpu()), dim=0)
                t_outputs_all = torch.cat((t_outputs_all, logits.cpu()), dim=0)
        
    # 计算准确率
    acc = n_correct / n_total if n_total > 0 else 0
    avg_loss = sum_loss / sum_step if sum_step > 0 else 0
    
    # 计算F1分数
    from sklearn.metrics import f1_score
    predictions = torch.argmax(t_outputs_all, dim=1)
    if macro:
        f1 = f1_score(t_targets_all.numpy(), predictions.numpy(), average='macro')
    else:
        f1 = f1_score(t_targets_all.numpy(), predictions.numpy(), average='micro')
    
    print(f"{mode} Accuracy: {acc:.4f}, Loss: {avg_loss:.4f}, F1: {f1:.4f}")
    
    return acc, f1, avg_loss


def test(args, model, test_data, device):
    """
    测试模型
    """
    # 获取processor
    processor = getattr(test_data, 'processor', None)
    
    # 确保模型加载了权重
    if not hasattr(model, 'loaded_weights'):
        # 尝试加载最佳模型
        model_path = os.path.join(args.save_dir, args.model_name, 'best_model.pt')
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"已加载最佳模型权重: {model_path}")
        else:
            print(f"警告: 未找到模型权重文件: {model_path}")
    
    # 移动模型到设备
    model.to(device)
    
    # 评估模型
    acc_macro, f1_macro, precision_macro, recall_macro = evaluate_acc_f1(
        args, model, device, test_data, processor, macro=True, mode='test'
    )
    acc_micro, f1_micro, precision_micro, recall_micro = evaluate_acc_f1(
        args, model, device, test_data, processor, macro=False, mode='test'
    )
    
    # 保存结果
    results = {
        "accuracy": acc_macro,
        "f1_macro": f1_macro,
        "f1_micro": f1_micro,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "precision_micro": precision_micro,
        "recall_micro": recall_micro,
        "model": args.model_name,
        "dataset": args.dataset
    }
    
    # 创建结果保存目录
    result_dir = os.path.join(args.save_dir, args.model_name)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    
    # 保存结果到文件
    result_file = os.path.join(result_dir, f"test_results_{time.strftime('%Y%m%d_%H%M%S')}.json")
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    
    # 打印测试结果
    print(f"\n测试结果 ({args.model_name}):")
    print(f"准确率 (宏平均): {acc_macro:.4f}")
    print(f"F1分数 (宏平均): {f1_macro:.4f}")
    print(f"精确率 (宏平均): {precision_macro:.4f}")
    print(f"召回率 (宏平均): {recall_macro:.4f}")
    print(f"F1分数 (微平均): {f1_micro:.4f}")
    print(f"精确率 (微平均): {precision_micro:.4f}")
    print(f"召回率 (微平均): {recall_micro:.4f}")
    print(f"结果已保存到: {result_file}")
    
    return results