from transformers import CLIPModel,BertConfig
from transformers.models.bert.modeling_bert import BertLayer
import torch.nn as nn
import torch
import torch.nn.functional as F
import copy

class MultimodalEncoder(nn.Module): # 本质上是“把 BERT 的 Transformer Encoder Layer 叠 N 层”，并在前向时可选地返回每一层的输出以及注意力矩阵
    def __init__(self, config, layer_number):
        super(MultimodalEncoder, self).__init__()
        layer = BertLayer(config) # 用 Hugging Face 的 BertLayer 创建一个标准的 Transformer encoder layer（包含自注意力 + 前馈网络 + 残差 + LayerNorm），其维度/头数等由 config 决定。
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(layer_number)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        all_encoder_layers = []
        all_encoder_attentions = []
        for layer_module in self.layer:
            hidden_states, attention = layer_module(hidden_states, attention_mask, output_attentions=True)
            all_encoder_attentions.append(attention)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers, all_encoder_attentions


class MV_CLIP(nn.Module):
    def __init__(self, args):
        super(MV_CLIP, self).__init__()
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32") # 从 Hugging Face 加载预训练的 CLIP 模型（ViT-B/32）。它负责把图像/文本编码成向量特征
        self.config = BertConfig.from_pretrained("bert-base-uncased") # 读取一份 BERT 的配置对象 BertConfig ，这里主要是“借用 BERT 的 Transformer 配置结构”
        self.config.hidden_size = 512 # 把 Transformer 的隐藏层维度改成 512，用来对齐 CLIP 的特征维度（CLIP ViT-B/32 的 embedding 通常是 512）。
        self.config.num_attention_heads = 8 # 设置多头注意力的头数为 8。要求 hidden_size 能被头数整除（512/8=64），这样每个 head 的维度是 64
        self.trans = MultimodalEncoder(self.config, layer_number=args.layers) # 用上面这份配置创建一个自定义的多模态 Transformer 编码器
        if args.simple_linear:
            self.text_linear =  nn.Linear(args.text_size, args.text_size)
            self.image_linear =  nn.Linear(args.image_size, args.image_size)
        else:
            self.text_linear =  nn.Sequential(
                nn.Linear(args.text_size, args.text_size),
                nn.Dropout(args.dropout_rate),
                nn.GELU()
            )
            self.image_linear =  nn.Sequential(
                nn.Linear(args.image_size, args.image_size),
                nn.Dropout(args.dropout_rate),
                nn.GELU()
            )

        self.classifier_fuse = nn.Linear(args.text_size , args.label_number)
        self.classifier_text = nn.Linear(args.text_size, args.label_number)
        self.classifier_image = nn.Linear(args.image_size, args.label_number)

        # 可学习权重
        self.cim_text_proj = nn.Linear(args.text_size, args.text_size, bias=False)
        self.cim_image_proj = nn.Linear(args.text_size, args.text_size, bias=False)
        self.cim_text_ln = nn.LayerNorm(args.text_size)
        self.cim_image_ln = nn.LayerNorm(args.text_size)
        self.cim_logit_scale = nn.Parameter(torch.tensor(0.0))
        self.fim_top_k = getattr(args, "fim_top_k", 5)

        # 两层 MLP（ d -> 4d -> d ，GELU）
        self.fim_text_ffn = nn.Sequential( # 对 文本特征 做前馈网络变换，和 Transformer 里标准的 FFN 结构一模一样（先升维、非线性、再降维），提升表达能力。
            nn.Linear(args.text_size, args.text_size * 4), # 扩展层
            nn.GELU(), # 激活函数
            nn.Linear(args.text_size * 4, args.text_size), # 压缩层
        )
        self.fim_image_ffn = nn.Sequential( # 对 视觉特征 做前馈网络变换
            nn.Linear(args.text_size, args.text_size * 4),
            nn.GELU(),
            nn.Linear(args.text_size * 4, args.text_size),
        )

        # 对文本 / 图像各自的特征做 LayerNorm
        self.fim_text_ln = nn.LayerNorm(args.text_size)
        self.fim_image_ln = nn.LayerNorm(args.text_size)

        self.fim_dynrt_iters = getattr(args, "fim_dynrt_iters", 3)

        self.fim_dynrt_image_value = nn.Linear(args.text_size, args.text_size, bias=False)
        self.fim_dynrt_text_value = nn.Linear(args.text_size, args.text_size, bias=False)

        self.fim_extra_fuse = nn.Linear(args.text_size * 2, args.text_size)

        self.sim_top_k = getattr(args, "sim_top_k", getattr(args, "fim_top_k", 5))
        self.sim_num_layers = getattr(args, "sim_num_layers", 1)
        self.sim_num_heads = getattr(args, "sim_num_heads", 4)
        if self.sim_num_heads < 1:
            self.sim_num_heads = 1
        if args.text_size % self.sim_num_heads != 0:
            self.sim_num_heads = 1
        self.sim_head_dim = args.text_size // self.sim_num_heads
        self.sim_q = nn.Linear(args.text_size, args.text_size, bias=False)
        self.sim_k = nn.Linear(args.text_size, args.text_size, bias=False)
        self.sim_v = nn.Linear(args.text_size, args.text_size, bias=False)
        self.sim_att_ln = nn.LayerNorm(args.text_size)
        self.sim_ffn = nn.Sequential(
            nn.Linear(args.text_size, args.text_size * 4),
            nn.GELU(),
            nn.Linear(args.text_size * 4, args.text_size),
        )
        self.sim_ffn_ln = nn.LayerNorm(args.text_size)
        self.sim_out = nn.Sequential(
            nn.Linear(args.text_size * 3, args.text_size),
            nn.GELU(),
        )
        self.sim_extra_fuse = nn.Linear(args.text_size * 3, args.text_size)
        self.sim_text_pool = nn.Linear(args.text_size, 1, bias=False)
        self.sim_image_pool = nn.Linear(args.text_size, 1, bias=False)

        self.loss_fct = nn.CrossEntropyLoss() # 交叉熵损失
        self.att = nn.Linear(args.text_size, 1, bias=False) # 一个线性层，把 hidden 向量映射成一个标量（logit），用于计算注意力权重



    def _dynrt_squash(self, x):
        squared_norm = (x * x).sum(dim=-1, keepdim=True)
        scale = squared_norm / (1.0 + squared_norm)
        x_norm = torch.sqrt(squared_norm + 1e-8)
        return scale * (x / x_norm)

    def _dynrt_route(self, u, b, iters):
        iters = int(iters)
        if iters < 1:
            iters = 1
        for i in range(iters):
            c = F.softmax(b, dim=-1)
            s = (c.unsqueeze(-1) * u).sum(dim=-2)
            v = self._dynrt_squash(s)
            if i < iters - 1:
                b = b + (u * v.unsqueeze(-2)).sum(dim=-1)
        return v

    # 图构建，根据文本 token、图像 region（或 patch）之间的相似度 interaction ，构造一个“图的邻接矩阵 mask”（ [bsz, m+n, m+n] 的 bool 张量）
    # m ：文本节点数（text tokens 数）
    # n ：视觉节点数（image regions/patches 数）
    # interaction ：形状大概率是 [bsz, m, n] ，表示每个文本 token 对每个视觉节点的相似度/匹配分数
    # text_attention_mask ：形状 [bsz, m] ，文本哪些 token 有效（padding 为 0）
    def _build_sim_graph_mask(self, text_attention_mask, interaction, m, n): 
        device = interaction.device
        bsz = interaction.shape[0]

        # 1）选 Top-K 跨模态边，得到两类跨模态稀疏连接：每个文本连到若干视觉、每个视觉连到若干文本。
        k_img = min(int(self.sim_top_k), n)
        if k_img < 1:
            k_img = 1
        topk_img = interaction.topk(k_img, dim=-1).indices # 对每个文本 token（最后一维是n）选出相似度最高的 k_img 个视觉节点索引。结果形状是 [bsz, m, k_img] 。
        interaction_t = interaction.transpose(1, 2) # 转置后，变成 [bsz, n, m] ，表示每个视觉节点对每个文本 token 的相似度
        k_txt = min(int(self.sim_top_k), m)
        if k_txt < 1:
            k_txt = 1
        topk_txt = interaction_t.topk(k_txt, dim=-1).indices # 对每个视觉节点（最后一维是m）选出相似度最高的 k_txt 个文本 token 索引。结果形状是 [bsz, n, k_txt] 。

        # 2）初始化整图 mask，并加“模态内”基础连接（每个文本连到前一个/后一个文本，每个视觉连到前一个/后一个视觉）
        l_total = m + n
        graph_mask = torch.zeros((bsz, l_total, l_total), dtype=torch.bool, device=device) 
        
        # 文本子图（前 m 个节点）加入“链式结构 + 自环”
        idx = torch.arange(m, device=device)
        idx_prev = idx - 1
        valid_prev = idx_prev >= 0
        graph_mask[:, idx[valid_prev], idx_prev[valid_prev]] = True # 有前驱则连前驱
        graph_mask[:, idx, idx] = True # 自环
        idx_next = idx + 1
        valid_next = idx_next < m
        graph_mask[:, idx[valid_next], idx_next[valid_next]] = True # 有后继则连后继

        graph_mask[:, m:, m:] = True # 视觉子图，视觉节点之间全连接（视觉内部任意两节点都可见）

        # 3) 加入跨模态 Top-K 连接
        t2v = torch.zeros((bsz, m, n), dtype=torch.bool, device=device) # [bsz, m, n] 的 bool，初始全 False
        t2v.scatter_(-1, topk_img, True) # 把每个文本 token 对应的 Top-K 视觉位置标 True。
        graph_mask[:, :m, m:] |= t2v # 将这些边写入总图的“文本->视觉”块

        v2t = torch.zeros((bsz, n, m), dtype=torch.bool, device=device) # [bsz, n, m] 的 bool，初始全 False
        v2t.scatter_(-1, topk_txt, True) # 把每个视觉节点 对应的 Top-K 文本 token 位置标 True。
        graph_mask[:, m:, :m] |= v2t # 将这些边写入总图的“视觉->文本”块

        # 4) 用有效节点 mask 清理 padding 的文本节点
        node_valid = torch.cat(
            (text_attention_mask.to(torch.bool), torch.ones((bsz, n), dtype=torch.bool, device=device)),
            dim=1,
        ) # 文本节点按 text_attention_mask 判定是否有效；视觉节点全部认为有效
        graph_mask = graph_mask & node_valid.unsqueeze(1) & node_valid.unsqueeze(2) # - 同时屏蔽掉：从无效节点出发的边（行方向）和指向无效节点的边（列方向），最终保证 padding token 不参与图计算。
        return graph_mask

    def _sim_graph_propagate(self, x, graph_mask): 
        h = x
        bsz, l_total, _ = h.shape
        mask = graph_mask.unsqueeze(1)
        for _ in range(self.sim_num_layers):
            q = self.sim_q(h)
            k = self.sim_k(h)
            v = self.sim_v(h)
            q = q.view(bsz, l_total, self.sim_num_heads, self.sim_head_dim).transpose(1, 2)
            k = k.view(bsz, l_total, self.sim_num_heads, self.sim_head_dim).transpose(1, 2)
            v = v.view(bsz, l_total, self.sim_num_heads, self.sim_head_dim).transpose(1, 2)
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) * (self.sim_head_dim ** -0.5)
            attn_scores = attn_scores.masked_fill(~mask, -1e4)
            attn_probs = F.softmax(attn_scores, dim=-1)
            attn_out = torch.matmul(attn_probs, v)
            attn_out = attn_out.transpose(1, 2).contiguous().view(bsz, l_total, -1)
            h = self.sim_att_ln(h + attn_out)
            h = self.sim_ffn_ln(h + self.sim_ffn(h))
        return h

    def forward(self, inputs, labels):
        output = self.model(**inputs,output_attentions=True)
        text_features = output['text_model_output']['last_hidden_state'] # 文本特征
        image_features = output['vision_model_output']['last_hidden_state'] # 图像特征
        text_feature = output['text_model_output']['pooler_output'] # 文本池化特征
        image_feature = output['vision_model_output']['pooler_output'] # 图像池化特征
        text_feature = self.text_linear(text_feature) # 文本特征线性变换
        image_feature = self.image_linear(image_feature) # 图像特征线性变换

        text_embeds = self.model.text_projection(text_features) # 文本特征投影 (B, m, d) = T
        image_embeds = self.model.visual_projection(image_features) # 图像特征投影 (B, n, d) = V
        
        # image_token_len = image_embeds.shape[1]
        # input_embeds = torch.cat((image_embeds, text_embeds), dim=1) # 合并文本特征和图像特征成为多模态嵌入特征
        # attention_mask = torch.cat(
        #     (
        #         torch.ones(text_features.shape[0], image_token_len, device=text_features.device),
        #         inputs['attention_mask'],
        #     ),
        #     dim=-1,
        # ) # 合并文本特征和图像特征的注意力掩码
        # extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2) # 扩展掩码维度，适配Transformer注意力计算
        # extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # 将注意力掩码转换为与模型参数 相同的数据类型
        # extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0 # 将注意力掩码转换为 注意力分数屏蔽值
        # fuse_hiddens, all_attentions = self.trans(input_embeds, extended_attention_mask, output_all_encoded_layers=False) # 多模态特征融合的核心步骤 ，通过自定义的Transformer编码器将图像和文本特征进行深度交互融合，输出融合后的隐藏状态和所有层的注意力矩阵。
        # fuse_hiddens = fuse_hiddens[-1] # 取 Transformer 编码器的最后一层输出作为融合后的隐藏状态
        # new_text_features = fuse_hiddens[:, image_token_len:, :] # 提取文本部分的融合特征

        # 公式2 张量形状（B=batch, m=文本长度, n=图像patch数, d=512）
        text_proj = self.cim_text_ln(F.normalize(self.cim_text_proj(text_embeds), p=2, dim=-1)) # (B, m, d) = (LN(TW_t))
        image_proj = self.cim_image_ln(F.normalize(self.cim_image_proj(image_embeds), p=2, dim=-1)) # (B, n, d) = (LN(VW_v))
        interaction = torch.matmul(text_proj, image_proj.transpose(1, 2)) * self.cim_logit_scale.exp() # (B, m, n) = 交互矩阵 (E)，已经乘上了温度 exp(cim_logit_scale)

        text_att_full = F.softmax(interaction, dim=-1)
        image_att_full = F.softmax(interaction.transpose(1, 2), dim=-1)
        text_c_full = torch.matmul(text_att_full, image_embeds)
        image_c_full = torch.matmul(image_att_full, text_embeds)

        # FIM 的 mask
        # 对每个文本 token i ，在 E[i, :] 上选 top‑k 的图像 patch
        k_img = min(int(self.fim_top_k), interaction.shape[-1])
        if k_img < 1:
            k_img = 1
        topk_img = interaction.topk(k_img, dim=-1).indices # (B, m, k)
        b_t2v = interaction.gather(dim=-1, index=topk_img)
        image_values = self.fim_dynrt_image_value(image_embeds)
        batch_index = torch.arange(image_values.shape[0], device=image_values.device)[:, None, None]
        u_t2v = image_values[batch_index, topk_img]
        text_dynrt = self._dynrt_route(u_t2v, b_t2v, self.fim_dynrt_iters) # 每个文本 token 聚合它最相关的若干图像 patch 的动态路由结果

        # 对每个图像 patch j ，在 E^T[j, :] 上选 top‑k 的文本 token
        interaction_t = interaction.transpose(1, 2) # (B, n, m)
        k_txt = min(int(self.fim_top_k), interaction_t.shape[-1])
        if k_txt < 1:
            k_txt = 1
        topk_txt = interaction_t.topk(k_txt, dim=-1).indices # (B, n, k)
        b_v2t = interaction_t.gather(dim=-1, index=topk_txt)
        text_values = self.fim_dynrt_text_value(text_embeds)
        u_v2t = text_values[batch_index, topk_txt]
        image_dynrt = self._dynrt_route(u_v2t, b_v2t, self.fim_dynrt_iters) # 每个图像 patch 聚合它最相关的若干文本 token 的动态路由结果

         # FIM 输出：这里用“非 mask 交互结果 - mask 交互结果”（残差）作为事实不一致信号
        # text_fim = text_c - text_c_masked # (B, m, d)
        # image_fim = image_c - image_c_masked # (B, n, d)

        # 用 “FFN + 残差 + LN” 得到 FIM 输出

        # 用 FIM 的动态路由结果更新文本 / 图像特征
        text_fim = self.fim_text_ln(text_embeds + self.fim_text_ffn(text_dynrt))
        image_fim = self.fim_image_ln(image_embeds + self.fim_image_ffn(image_dynrt))

        # 基础分支（base）：从最后一个文本 token 和图像 CLS 抽取全局特征并融合
        last_token_index = inputs['attention_mask'].to(torch.long).sum(dim=-1) - 1
        last_token_index = last_token_index.clamp(min=0)

        base_text_features = text_c_full
        base_text_feature = base_text_features[
            torch.arange(base_text_features.shape[0], device=base_text_features.device),
            last_token_index,
        ]
        base_image_feature = image_c_full[:, 0, :].squeeze(1)
        base_text_weight = self.att(base_text_feature) # 用打分器 self.att 对文本全局特征打分
        base_image_weight = self.att(base_image_feature) # 用打分器 self.att 对图像全局特征打分
        base_att = nn.functional.softmax(torch.stack((base_text_weight, base_image_weight), dim=-1),dim=-1) 
        base_tw, base_iw = base_att.split([1,1], dim=-1) # 把两路分数拼在一起做 softmax，得到“文本/图像之间的权重比例”
        base_fuse_feature = base_tw.squeeze(1) * base_text_feature + base_iw.squeeze(1) * base_image_feature # 加权融合

        # FIM 分支（fim）：对经过 FIM 更新的文本 / 图像再做一次同样的加权融合
        fim_text_features = text_fim
        fim_text_feature = fim_text_features[
            torch.arange(fim_text_features.shape[0], device=fim_text_features.device),
            last_token_index,
        ] # 从 text_fim 中取“每个样本最后一个真实 token 的向量”，形状 [B, d] 。
        fim_image_feature = image_fim[:, 0, :].squeeze(1) # 从 image_fim 取第 0 个位置作为图像全局向量 [B, d] 。
        fim_text_weight = self.att(fim_text_feature)
        fim_image_weight = self.att(fim_image_feature)
        fim_att = nn.functional.softmax(torch.stack((fim_text_weight, fim_image_weight), dim=-1),dim=-1)
        fim_tw, fim_iw = fim_att.split([1,1], dim=-1) # 同样用 self.att 打分并 softma
        fim_fuse_feature = fim_tw.squeeze(1) * fim_text_feature + fim_iw.squeeze(1) * fim_image_feature # 这是 “经过 FIM 交互和 FFN/LN 更新后的文本‑图像融合特征” 。

        m = text_embeds.shape[1]
        n = image_embeds.shape[1]
        graph_mask = self._build_sim_graph_mask(inputs['attention_mask'], interaction, m, n)
        sim_nodes = torch.cat((text_fim, image_fim), dim=1) # 图节点，(B, m+n, d)
        sim_nodes = self._sim_graph_propagate(sim_nodes, graph_mask)
        sim_text = sim_nodes[:, :m, :]
        sim_image = sim_nodes[:, m:, :]
        text_mask = inputs['attention_mask'].to(torch.bool)
        text_scores = self.sim_text_pool(sim_text).squeeze(-1)
        text_scores = text_scores.masked_fill(~text_mask, -1e4)
        text_alpha = F.softmax(text_scores, dim=-1).unsqueeze(-1)
        sim_text_feature = (sim_text * text_alpha).sum(dim=1)
        image_scores = self.sim_image_pool(sim_image).squeeze(-1)
        image_alpha = F.softmax(image_scores, dim=-1).unsqueeze(-1)
        sim_image_feature = (sim_image * image_alpha).sum(dim=1)
        sim_fuse_feature = self.sim_out(
            torch.cat((sim_text_feature, sim_image_feature, (sim_text_feature - sim_image_feature).abs()), dim=-1)
        )

        fuse_feature = self.sim_extra_fuse(torch.cat((base_fuse_feature, fim_fuse_feature, sim_fuse_feature), dim=-1))

        # 通过三个分类头得到未归一化的分类分数
        logits_fuse = self.classifier_fuse(fuse_feature)
        logits_text = self.classifier_text(text_feature)
        logits_image = self.classifier_image(image_feature)
   
        # 用 softmax 把 logits 变成每个类别的概率
        fuse_score = nn.functional.softmax(logits_fuse, dim=-1)
        text_score = nn.functional.softmax(logits_text, dim=-1)
        image_score = nn.functional.softmax(logits_image, dim=-1)

        score = fuse_score + text_score + image_score

        outputs = (score,)
        if labels is not None:
            loss_fuse = self.loss_fct(logits_fuse, labels)
            loss_text = self.loss_fct(logits_text, labels)
            loss_image = self.loss_fct(logits_image, labels)
            loss = loss_fuse + loss_text + loss_image

            outputs = (loss,) + outputs
        return outputs
