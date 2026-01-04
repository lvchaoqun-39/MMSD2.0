import os
from data_set import MyDataset
from torch.utils.data import DataLoader
import torch
import logging
from tqdm import tqdm, trange
from sklearn import metrics
import wandb
import numpy as np

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def train(args, model, device, train_data, dev_data, test_data, processor):
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)  # 如果输出目录不存在，就创建它

    # train_loader 是一个可迭代对象；遍历它时（ for step, batch in enumerate(iter_bar): ），每次得到的 batch 就是 collate_func 组装好的一个 batch： (text_list, image_list, label_list, id_list)
    train_loader = DataLoader(dataset=train_data,
                              batch_size=args.train_batch_size, # 分批
                              collate_fn=MyDataset.collate_func, # 将单个数据点处理成模型输入的格式
                              shuffle=True) # 每个epoch开始前随机打乱数据顺序
    total_steps = int(len(train_loader) * args.num_train_epochs) # 全程一共会跑多少个 batch
    model.to(device) # 把模型的参数和缓冲区（weights、bias、BatchNorm 的 running stats 等） 移动到指定计算设备上。

    # 优化器（optimizer）：根据当前的梯度，决定“模型参数要往哪个方向、走多大一步”来减小损失（loss）。
    if args.optimizer_name == 'adafactor': # 优化器和学习率调度器
        from transformers.optimization import Adafactor, AdafactorSchedule

        print('Use Adafactor Optimizer for Training.')
        optimizer = Adafactor(
            model.parameters(),
            # lr=1e-3,
            # eps=(1e-30, 1e-3),
            # clip_threshold=1.0,
            # decay_rate=-0.8,
            # beta1=None,
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
            clip_params = list(map(id, model.model.parameters())) # 取出 model.model 这部分参数对象的“身份标识”(内存地址层面的 id)
            base_params = filter(lambda p: id(p) not in clip_params, model.parameters()) # 剩下的就是“非 CLIP 主干”的参数（比如额外加的分类头、融合层等）。 这么做的原因是：如果不排除，CLIP 主干参数会同时出现在两个参数组里，导致重复更新/冲突。
            optimizer = AdamW([
                    {"params": base_params}, # 基础部分参数，使用默认学习率 args.learning_rate
                    {"params": model.model.parameters(),"lr": args.clip_learning_rate} # CLIP 主干参数，单独用更小/不同的学习率 args.clip_learning_rate（常见做法：微调预训练 backbone 时学习率更小）
                    ], lr=args.learning_rate, weight_decay=args.weight_decay)

            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(args.warmup_proportion * total_steps),
                                                    num_training_steps=total_steps) # 学习率会线性增加，从 warmup_proportion * total_steps 到 total_steps
        else:
            optimizer = optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon, weight_decay=args.weight_decay)
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(args.warmup_proportion * total_steps),
                                                num_training_steps=total_steps)
    else:
        raise Exception('Wrong Optimizer Name!!!')


    best_dev_f1 = -1.0
    best_dev_acc = 0.0
    for i_epoch in trange(0, int(args.num_train_epochs), desc="Epoch", disable=False):
        sum_loss = 0.
        sum_step = 0

        iter_bar = tqdm(train_loader, desc="Iter (loss=X.XXX)", disable=False) # 对 batch 循环加进度条
        model.train()

        for step, batch in enumerate(iter_bar): # 每次迭代完成一次“前向→算 loss→反向→参数更新”
            text_list, image_list, label_list, id_list = batch # 把当前 batch 解包成 4 份内容：文本、图像、标签、样本 id
            if args.model == 'MV_CLIP':
                inputs = processor(text=text_list, images=image_list, padding='max_length', truncation=True, max_length=args.max_len, return_tensors="pt").to(device)
                labels = torch.tensor(label_list).to(device)

            loss, score = model(inputs,labels=labels) # 前向计算：把 inputs 和 labels 喂给模型，返回损失 loss 和输出 score （通常是 logits/预测分数）。
            sum_loss += loss.item()
            sum_step += 1

            iter_bar.set_description("Iter (loss=%5.3f)" % loss.item()) # 更新进度条显示，把当前 batch 的 loss 动态写到进度条标题里。
            loss.backward() # 反向传播
            optimizer.step() # 更新参数
            if args.optimizer_name == 'adam':
                scheduler.step() # 仅当使用 Adam 分支时推进学习率调度器，让学习率按 warmup/衰减策略变化。
            optimizer.zero_grad() # 清空梯度，为下一个 batch 做准备

        wandb.log({'train_loss': sum_loss/sum_step})
        dev_acc, dev_f1 ,dev_precision,dev_recall = evaluate_acc_f1(args, model, device, dev_data, processor, mode='dev')
        wandb.log({'dev_acc': dev_acc, 'dev_f1': dev_f1, 'dev_precision': dev_precision, 'dev_recall': dev_recall})
        logging.info("i_epoch is {}, dev_acc is {}, dev_f1 is {}, dev_precision is {}, dev_recall is {}".format(i_epoch, dev_acc, dev_f1, dev_precision, dev_recall))

        if dev_f1 > best_dev_f1 or (dev_f1 == best_dev_f1 and dev_acc > best_dev_acc):
            best_dev_f1 = dev_f1
            best_dev_acc = dev_acc

            path_to_save = os.path.join(args.output_dir, args.model)
            if not os.path.exists(path_to_save):
                os.mkdir(path_to_save)
            model_to_save = (model.module if hasattr(model, "module") else model)
            torch.save(model_to_save.state_dict(), os.path.join(path_to_save, 'model.pt'))

            test_acc, test_f1,test_precision,test_recall = evaluate_acc_f1(args, model, device, test_data, processor,macro = True, mode='test')
            _, test_f1_,test_precision_,test_recall_ = evaluate_acc_f1(args, model, device, test_data, processor, mode='test')
            wandb.log({'test_acc': test_acc, 'macro_test_f1': test_f1,
                     'macro_test_precision': test_precision,'macro_test_recall': test_recall, 'micro_test_f1': test_f1_,
                     'micro_test_precision': test_precision_,'micro_test_recall': test_recall_})
            logging.info("i_epoch is {}, test_acc is {}, macro_test_f1 is {}, macro_test_precision is {}, macro_test_recall is {}, micro_test_f1 is {}, micro_test_precision is {}, micro_test_recall is {}".format(i_epoch, test_acc, test_f1, test_precision, test_recall, test_f1_, test_precision_, test_recall_))
            # test_loss : 测试损失（衡量测试结果和真实答案之间的差距） 。
            # test_acc : 测试集准确率（无论是否讽刺，判断正确的比例） 。这是衡量模型最终泛化能力的核心指标。
            # macro_test_f1 / micro_test_f1 : 测试集 F1-Score (宏平均和微平均)（f1是precision和recall的调和平均值）。
            # macro_test_precision / micro_test_precision : 测试集精确率（找出来是讽刺的找的准不准） 。
            # macro_test_recall / micro_test_recall : 测试集召回率（找的全不全） 。

        torch.cuda.empty_cache()
    logger.info('Train done')


def evaluate_acc_f1(args, model, device, data, processor, macro=False,pre = None, mode='test'):
        data_loader = DataLoader(data, batch_size=args.dev_batch_size, collate_fn=MyDataset.collate_func,shuffle=False)
        n_correct, n_total = 0, 0 # n_total 当前已经累计的样本总数； n_total 当前已经累计的 样本总数
        t_targets_all, t_outputs_all = None, None # t_targets_all 整个数据集所有样本的真实标签；t_outputs_all 整个数据集所有样本的预测标签

        model.eval() # 把模型切换到评估模式，而不是训练模式
        sum_loss = 0.
        sum_step = 0
        with torch.no_grad(): # 关闭梯度计算：评估时不需要反向传播，可以省显存、加快速度
            for i_batch, t_batch in enumerate(data_loader):
                text_list, image_list, label_list, id_list = t_batch
                if args.model == 'MV_CLIP':
                    # 用 processor 把文本+图像处理成模型输入张量 inputs ，并把 labels 转成张量，都放到 device 上。
                    inputs = processor(text=text_list, images=image_list, padding='max_length', truncation=True, max_length=args.max_len, return_tensors="pt").to(device)
                    labels = torch.tensor(label_list).to(device)
                
                t_targets = labels # 把真实标签保存为 t_targets
                loss, t_outputs = model(inputs,labels=labels)
                sum_loss += loss.item()
                sum_step += 1
  
                outputs = torch.argmax(t_outputs, -1) # 把分数转换成预测类别：对最后一维（类别维）取最大值的索引，得到每个样本的预测标签

                n_correct += (outputs == t_targets).sum().item() # 计算正确预测的样本数量
                n_total += len(outputs) # 计算总样本数

                # 把所有 batch 的预测和标签拼起来，便于最后一次性算 F1/Precision/Recall
                if t_targets_all is None:
                    t_targets_all = t_targets
                    t_outputs_all = outputs
                else:
                    t_targets_all = torch.cat((t_targets_all, t_targets), dim=0)
                    t_outputs_all = torch.cat((t_outputs_all, outputs), dim=0)
        if mode == 'test':
            wandb.log({'test_loss': sum_loss/sum_step})
        else:
            wandb.log({'dev_loss': sum_loss/sum_step})
        if pre != None:
            with open(pre,'w',encoding='utf-8') as fout:
                predict = t_outputs_all.cpu().numpy().tolist() # 把预测标签转换成普通列表
                label = t_targets_all.cpu().numpy().tolist() # 把真实标签转换成普通列表
                for x,y,z in zip(predict,label):
                    fout.write(str(x) + str(y) +z+ '\n')
        if not macro:   # macro=False ：更像是“看正类(通常是 1)的 P/R/F1”
            acc = n_correct / n_total
            f1 = metrics.f1_score(t_targets_all.cpu(), t_outputs_all.cpu())
            precision =  metrics.precision_score(t_targets_all.cpu(),t_outputs_all.cpu())
            recall = metrics.recall_score(t_targets_all.cpu(),t_outputs_all.cpu())
        else:  # macro=True ：更像“宏 F1”（宏 F1 是 F1 的宏版本，考虑了正负样本的数量，对 F1 进行归一化处理）
            acc = n_correct / n_total
            f1 = metrics.f1_score(t_targets_all.cpu(), t_outputs_all.cpu(), labels=[0, 1],average='macro')
            precision =  metrics.precision_score(t_targets_all.cpu(),t_outputs_all.cpu(), labels=[0, 1],average='macro')
            recall = metrics.recall_score(t_targets_all.cpu(),t_outputs_all.cpu(), labels=[0, 1],average='macro')
        return acc, f1 ,precision,recall
