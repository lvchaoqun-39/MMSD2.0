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
        self.fim_text_ffn = nn.Sequential(
            nn.Linear(args.text_size, args.text_size * 4), # 扩展层
            nn.GELU(), # 激活函数
            nn.Linear(args.text_size * 4, args.text_size), # 压缩层
        )
        self.fim_image_ffn = nn.Sequential(
            nn.Linear(args.text_size, args.text_size * 4),
            nn.GELU(),
            nn.Linear(args.text_size * 4, args.text_size),
        )
        self.fim_text_ln = nn.LayerNorm(args.text_size)
        self.fim_image_ln = nn.LayerNorm(args.text_size)

        self.fim_dynrt_iters = getattr(args, "fim_dynrt_iters", 3)
        self.fim_dynrt_image_value = nn.Linear(args.text_size, args.text_size, bias=False)
        self.fim_dynrt_text_value = nn.Linear(args.text_size, args.text_size, bias=False)
        self.fim_extra_fuse = nn.Linear(args.text_size * 2, args.text_size)

        self.loss_fct = nn.CrossEntropyLoss()
        self.att = nn.Linear(args.text_size, 1, bias=False)

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
        text_dynrt = self._dynrt_route(u_t2v, b_t2v, self.fim_dynrt_iters)

        # 对每个图像 patch j ，在 E^T[j, :] 上选 top‑k 的文本 token
        interaction_t = interaction.transpose(1, 2) # (B, n, m)
        k_txt = min(int(self.fim_top_k), interaction_t.shape[-1])
        if k_txt < 1:
            k_txt = 1
        topk_txt = interaction_t.topk(k_txt, dim=-1).indices # (B, n, k)
        b_v2t = interaction_t.gather(dim=-1, index=topk_txt)
        text_values = self.fim_dynrt_text_value(text_embeds)
        u_v2t = text_values[batch_index, topk_txt]
        image_dynrt = self._dynrt_route(u_v2t, b_v2t, self.fim_dynrt_iters)

         # FIM 输出：这里用“非 mask 交互结果 - mask 交互结果”（残差）作为事实不一致信号
        # text_fim = text_c - text_c_masked # (B, m, d)
        # image_fim = image_c - image_c_masked # (B, n, d)

        # 用 “FFN + 残差 + LN” 得到 FIM 输出
        text_fim = self.fim_text_ln(text_embeds + self.fim_text_ffn(text_dynrt))
        image_fim = self.fim_image_ln(image_embeds + self.fim_image_ffn(image_dynrt))

        last_token_index = inputs['attention_mask'].to(torch.long).sum(dim=-1) - 1
        last_token_index = last_token_index.clamp(min=0)

        base_text_features = text_c_full
        base_text_feature = base_text_features[
            torch.arange(base_text_features.shape[0], device=base_text_features.device),
            last_token_index,
        ]
        base_image_feature = image_c_full[:, 0, :].squeeze(1)
        base_text_weight = self.att(base_text_feature)
        base_image_weight = self.att(base_image_feature)
        base_att = nn.functional.softmax(torch.stack((base_text_weight, base_image_weight), dim=-1),dim=-1)
        base_tw, base_iw = base_att.split([1,1], dim=-1)
        base_fuse_feature = base_tw.squeeze(1) * base_text_feature + base_iw.squeeze(1) * base_image_feature

        fim_text_features = text_fim
        fim_text_feature = fim_text_features[
            torch.arange(fim_text_features.shape[0], device=fim_text_features.device),
            last_token_index,
        ]
        fim_image_feature = image_fim[:, 0, :].squeeze(1)
        fim_text_weight = self.att(fim_text_feature)
        fim_image_weight = self.att(fim_image_feature)
        fim_att = nn.functional.softmax(torch.stack((fim_text_weight, fim_image_weight), dim=-1),dim=-1)
        fim_tw, fim_iw = fim_att.split([1,1], dim=-1)
        fim_fuse_feature = fim_tw.squeeze(1) * fim_text_feature + fim_iw.squeeze(1) * fim_image_feature

        fuse_feature = self.fim_extra_fuse(torch.cat((base_fuse_feature, fim_fuse_feature), dim=-1))

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
