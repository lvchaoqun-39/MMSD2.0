import os
import torch
import logging
from tqdm import tqdm, trange
from sklearn import metrics
from data_set import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LinearLR
import wandb
import numpy as np

# 配置logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 添加Focal Loss类
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        # 计算交叉熵损失
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # 计算pt值
        pt = torch.exp(-ce_loss)
        
        # 应用Focal Loss公式
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        # 按照指定的方式进行损失聚合
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def train(args, model, device, train_dataset, valid_dataset, test_dataset, tokenizer):
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)  # 如果输出目录不存在，就创建它

    # train_loader 是一个可迭代对象；遍历它时（ for step, batch in enumerate(iter_bar): ），每次得到的 batch 就是 collate_func 组装好的一个 batch： (text_list, image_list, label_list, id_list)
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=args.train_batch_size, # 分批
                              collate_fn=MyDataset.collate_func, # 将单个数据点处理成模型输入的格式
                              shuffle=True) # 每个epoch开始前随机打乱数据顺序
    total_steps = int(len(train_loader) * args.num_train_epochs) # 全程一共会跑多少个 batch
    model.to(device) # 把模型的参数和缓冲区（weights、bias、BatchNorm 的 running stats 等） 移动到指定计算设备上。

    # 优化器（optimizer）：根据当前的梯度，决定"模型参数要往哪个方向、走多大一步"来减小损失（loss）。
    if args.optimizer_name == 'adafactor': # 优化器和学习率调度器
        from transformers.optimization import Adafactor, AdafactorSchedule

        print('Use Adafactor Optimizer for Training.')
        optimizer = Adafactor(
            model.parameters(),
            lr=None, # 学习率
            weight_decay=args.weight_decay, # 权重衰减（防止过拟合）
            relative_step=True, # 相对步数
            scale_parameter=True, # 缩放参数
            warmup_init=True , # 预热初始化
        )
        scheduler = AdafactorSchedule(optimizer) # 创建与 Adafactor 配套的学习率调度器
    elif args.optimizer_name == 'adam':
        print('Use AdamW Optimizer for Training.')
        from transformers.optimization import AdamW, get_linear_schedule_with_warmup
        if args.model == 'MV_CLIP':
            clip_params = list(map(id, model.model.parameters())) # 取出 model.model 这部分参数对象的"身份标识"(内存地址层面的 id)
            base_params = filter(lambda p: id(p) not in clip_params, model.parameters()) # 剩下的就是"非 CLIP 主干"的参数（比如额外加的分类头、融合层等）。 这么做的原因是：如果不排除，CLIP 主干参数会同时出现在两个参数组里，导致重复更新/冲突。
            optimizer = AdamW([
                    {"params": base_params}, # 基础部分参数，使用默认学习率 args.learning_rate
                    {"params": model.model.parameters(),"lr": args.clip_learning_rate} # CLIP 主干参数，单独用更小/不同的学习率 args.clip_learning_rate（常见做法：微调预训练 backbone 时学习率更小）
                    ], lr=args.learning_rate, weight_decay=args.weight_decay)

            # 实现余弦退火学习率策略
            if hasattr(args, 'use_cosine_schedule') and args.use_cosine_schedule:
                print('Using Cosine Annealing Learning Rate Schedule')
                from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
                # 计算预热步数
                warmup_steps = int(args.warmup_proportion * total_steps)
                # 创建预热调度器
                warmup_scheduler = get_linear_schedule_with_warmup(
                    optimizer, 
                    num_warmup_steps=warmup_steps,
                    num_training_steps=warmup_steps
                )
                # 创建余弦退火调度器
                scheduler = CosineAnnealingWarmRestarts(
                    optimizer, 
                    T_0=args.T_0,  # 初始重启周期
                    T_mult=args.T_mult,  # 重启周期的乘法因子
                    eta_min=args.min_learning_rate  # 最小学习率
                )
                use_cosine = True
            else:
                print('Using Linear Learning Rate Schedule')
                scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(args.warmup_proportion * total_steps),
                                                        num_training_steps=total_steps)
                use_cosine = False
        else:
            optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon, weight_decay=args.weight_decay)
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(args.warmup_proportion * total_steps),
                                                num_training_steps=total_steps)
            use_cosine = False
    else:
        raise Exception('Wrong Optimizer Name!!!')
    
    # 如果使用Focal Loss，更新模型的损失函数
    if hasattr(args, 'use_focal_loss') and args.use_focal_loss:
        model.loss_fct = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma).to(device)
        print('Using Focal Loss with alpha={}, gamma={}'.format(args.focal_alpha, args.focal_gamma))

    max_acc = 0.
    patience_counter = 0  # 早停计数器
    best_epoch = 0  # 最佳模型对应的epoch
    
    for i_epoch in trange(0, int(args.num_train_epochs), desc="Epoch", disable=False):
        sum_loss = 0.
        sum_step = 0

        iter_bar = tqdm(train_loader, desc="Iter (loss=X.XXX)", disable=False) # 对 batch 循环加进度条
        model.train()

        for step, batch in enumerate(iter_bar): # 每次迭代完成一次"前向→算 loss→反向→参数更新"
            text_list, image_list, label_list, id_list = batch # 把当前 batch 解包成 4 份内容：文本、图像、标签、样本 id
            if args.model == 'MV_CLIP':
                # 使用tokenizer处理文本，这里简化处理
                inputs = {}
                # 处理文本，确保不超过CLIP模型的最大输入长度
                text_inputs = tokenizer(text_list, padding='max_length', truncation=True, max_length=min(args.max_len, 77), return_tensors="pt")
                inputs['input_ids'] = text_inputs['input_ids'].to(device)
                inputs['attention_mask'] = text_inputs['attention_mask'].to(device)
                # 处理图像（假设image_list已经是tensor格式）
                inputs['pixel_values'] = torch.stack(image_list).to(device) if isinstance(image_list[0], torch.Tensor) else torch.tensor(image_list).to(device)
                labels = torch.tensor(label_list).to(device)

            loss, score = model(inputs, labels=labels) # 前向计算：把 inputs 和 labels 喂给模型，返回损失 loss 和输出 score （通常是 logits/预测分数）。
            sum_loss += loss.item()
            sum_step += 1

            iter_bar.set_description("Iter (loss=%5.3f)" % loss.item()) # 更新进度条显示，把当前 batch 的 loss 动态写到进度条标题里。
            loss.backward() # 反向传播
            optimizer.step() # 更新参数
            
            # 学习率调度器步进
            if args.optimizer_name == 'adam':
                current_step = i_epoch * len(train_loader) + step
                if use_cosine:
                    # 先执行预热
                    if hasattr(args, 'use_cosine_schedule') and args.use_cosine_schedule:
                        warmup_steps = int(args.warmup_proportion * total_steps)
                        if current_step < warmup_steps:
                            warmup_scheduler.step()
                        else:
                            # 预热后切换到余弦退火
                            scheduler.step()
                    else:
                        scheduler.step()
                else:
                    scheduler.step()
            
            optimizer.zero_grad() # 清空梯度，为下一个 batch 做准备
            
            # 记录学习率
            if step % 100 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                wandb.log({'learning_rate': current_lr})

        wandb.log({'train_loss': sum_loss/sum_step})
        dev_acc, dev_f1 ,dev_precision,dev_recall = evaluate_acc_f1(args, model, device, valid_dataset, tokenizer, mode='dev')
        wandb.log({'dev_acc': dev_acc, 'dev_f1': dev_f1, 'dev_precision': dev_precision, 'dev_recall': dev_recall})
        logging.info("i_epoch is {}, dev_acc is {}, dev_f1 is {}, dev_precision is {}, dev_recall is {}".format(i_epoch, dev_acc, dev_f1, dev_precision, dev_recall))

        # 早停机制逻辑
        if dev_acc > max_acc:
            max_acc = dev_acc
            best_epoch = i_epoch
            patience_counter = 0  # 重置计数器

            path_to_save = os.path.join(args.output_dir, args.model)
            if not os.path.exists(path_to_save):
                os.mkdir(path_to_save)
            model_to_save = (model.module if hasattr(model, "module") else model)
            torch.save(model_to_save.state_dict(), os.path.join(path_to_save, 'model.pt'))

            test_acc, test_f1,test_precision,test_recall = evaluate_acc_f1(args, model, device, test_dataset, tokenizer, macro=True, mode='test')
            _, test_f1_,test_precision_,test_recall_ = evaluate_acc_f1(args, model, device, test_dataset, tokenizer, mode='test')
            wandb.log({'test_acc': test_acc, 'macro_test_f1': test_f1,
                     'macro_test_precision': test_precision,'macro_test_recall': test_recall, 'micro_test_f1': test_f1_,
                     'micro_test_precision': test_precision_,'micro_test_recall': test_recall_})
            logging.info("i_epoch is {}, test_acc is {}, macro_test_f1 is {}, macro_test_precision is {}, macro_test_recall is {}, micro_test_f1 is {}, micro_test_precision is {}, micro_test_recall is {}".format(i_epoch, test_acc, test_f1, test_precision, test_recall, test_f1_, test_precision_, test_recall_))
        else:
            patience_counter += 1
            logging.info("Patience counter: {}/{}".format(patience_counter, args.patience))
            
            # 触发早停
            if hasattr(args, 'patience') and patience_counter >= args.patience:
                logging.info("Early stopping triggered! Best epoch: {}".format(best_epoch))
                break

        torch.cuda.empty_cache()
    logger.info('Train done')

def evaluate_acc_f1(args, model, device, dataset, tokenizer, mode='dev'):
    # 创建数据加载器，使用dataset的collate_func方法
    batch_size = args.test_batch_size if hasattr(args, 'test_batch_size') else args.batch_size
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=dataset.collate_func
    )
    
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for inputs in tqdm(dataloader, desc=f"Evaluating {mode}"):
            # 将inputs中的数据移到device上
            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels = inputs['labels']
            
            # 解包inputs字典并调用模型
            loss, t_outputs = model(**inputs)
            total_loss += loss.item()
            
            # 计算预测结果
            predictions = torch.argmax(t_outputs, dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 计算指标
    avg_loss = total_loss / len(dataloader)
    accuracy = metrics.accuracy_score(all_labels, all_predictions)
    f1_score = metrics.f1_score(all_labels, all_predictions, average='macro')
    precision = metrics.precision_score(all_labels, all_predictions, average='macro')
    recall = metrics.recall_score(all_labels, all_predictions, average='macro')
    
    # 记录结果到wandb
    if args.use_wandb:
        import wandb
        # 检查wandb是否已经初始化
        if wandb.run is not None:
            wandb.log({
                f"{mode}_loss": avg_loss,
                f"{mode}_accuracy": accuracy,
                f"{mode}_f1": f1_score,
                f"{mode}_precision": precision,
                f"{mode}_recall": recall
            })
    
    # 打印评估结果
    print(f"{mode} Loss: {avg_loss:.4f}")
    print(f"{mode} Accuracy: {accuracy:.4f}")
    print(f"{mode} F1 Score: {f1_score:.4f}")
    print(f"{mode} Precision: {precision:.4f}")
    print(f"{mode} Recall: {recall:.4f}")
    
    return accuracy, f1_score, precision, recall


def train(args, model, device, train_dataset, dev_dataset, optimizer, scheduler, tokenizer):
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=train_dataset.collate_func
    )
    
    best_f1 = 0
    patience = 0
    for epoch in range(args.num_train_epochs):
        logger.info(f'Epoch {epoch + 1}/{args.num_train_epochs}')
        
        # 训练模式
        model.train()
        total_loss = 0
        
        for step, inputs in enumerate(tqdm(train_loader, desc="Training")):
            # 将inputs中的数据移到device上
            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels = inputs['labels']
            
            # 清零梯度
            optimizer.zero_grad()
            
            # 前向传播
            loss, outputs = model(**inputs)
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            if hasattr(args, 'max_grad_norm') and args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            
            # 更新参数
            optimizer.step()
            
            # 更新学习率调度器
            if scheduler is not None:
                scheduler.step()
            
            total_loss += loss.item()
            
            # 记录到wandb
            if args.use_wandb and hasattr(args, 'logging_steps') and step % args.logging_steps == 0:
                wandb.log({
                    "train_loss": loss.item(),
                    "learning_rate": optimizer.param_groups[0]['lr']
                })
        
        # 计算平均训练损失
        avg_train_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1} - Average Train Loss: {avg_train_loss:.4f}")
        
        # 评估
        dev_acc, dev_f1, dev_precision, dev_recall = evaluate_acc_f1(
            args, model, device, dev_dataset, tokenizer, mode='dev'
        )
        
        # 早停机制
        if dev_f1 > best_f1:
            best_f1 = dev_f1
            patience = 0
            # 保存最好的模型
            if hasattr(args, 'save_model') and args.save_model:
                torch.save(model.state_dict(), os.path.join(args.output_dir, 'best_model.bin'))
            logger.info(f"New best F1: {best_f1:.4f} - Model saved!")
        else:
            patience += 1
            if patience >= args.early_stop:
                logger.info(f"Early stopping triggered after {patience} epochs without improvement")
                break
    
    return best_f1