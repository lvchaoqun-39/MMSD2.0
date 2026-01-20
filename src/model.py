from transformers import CLIPModel,BertConfig,RobertaModel,ViTModel
from transformers.models.bert.modeling_bert import BertLayer
import torch.nn as nn
import torch
import torch.nn.functional as F
import copy

torch.cuda.empty_cache()


class BipartiteGraphLayer(nn.Module):
    def __init__(self, hidden_size: int, dropout_rate: float, use_global: bool):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.use_global = bool(use_global)

        self.text_value = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.image_value = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        self.text_ffn = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size * 4),
            nn.GELU(),
            nn.Linear(self.hidden_size * 4, self.hidden_size),
        )
        self.image_ffn = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size * 4),
            nn.GELU(),
            nn.Linear(self.hidden_size * 4, self.hidden_size),
        )
        self.text_ln = nn.LayerNorm(self.hidden_size)
        self.image_ln = nn.LayerNorm(self.hidden_size)
        self.msg_dropout = nn.Dropout(dropout_rate)

        if self.use_global:
            self.global_from_nodes = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=False)
            self.global_to_text = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
            self.global_to_image = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
            self.global_ffn = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size * 4),
                nn.GELU(),
                nn.Linear(self.hidden_size * 4, self.hidden_size),
            )
            self.global_ln = nn.LayerNorm(self.hidden_size)

    def _masked_mean(self, x, mask):
        if mask is None:
            return x.mean(dim=1)
        mask = mask.to(dtype=x.dtype).unsqueeze(-1)
        denom = mask.sum(dim=1).clamp(min=1.0)
        return (x * mask).sum(dim=1) / denom

    def forward(
        self,
        text_states,
        image_states,
        interaction,
        attention_mask,
        top_k: int,
        edge_dropout: float,
        global_state=None,
    ):
        bsz, text_len, hidden = text_states.shape
        _, image_len, _ = image_states.shape

        if attention_mask is not None:
            text_valid = attention_mask.to(dtype=torch.bool)
        else:
            text_valid = None

        k_img = min(int(top_k), int(image_len))
        if k_img < 1:
            k_img = 1
        k_txt = min(int(top_k), int(text_len))
        if k_txt < 1:
            k_txt = 1

        topk_img = interaction.topk(k_img, dim=-1).indices
        scores_t2v = interaction.gather(dim=-1, index=topk_img)
        att_t2v = F.softmax(scores_t2v, dim=-1)
        if self.training and float(edge_dropout) > 0:
            att_t2v = F.dropout(att_t2v, p=float(edge_dropout), training=True)
            att_t2v = att_t2v / (att_t2v.sum(dim=-1, keepdim=True) + 1e-8)
        image_values = self.image_value(image_states)
        batch_index = torch.arange(bsz, device=image_values.device)[:, None, None]
        gathered_image_values = image_values[batch_index, topk_img]
        msg_text = (att_t2v.unsqueeze(-1) * gathered_image_values).sum(dim=-2)
        if text_valid is not None:
            msg_text = msg_text * text_valid.unsqueeze(-1).to(dtype=msg_text.dtype)

        interaction_t = interaction.transpose(1, 2)
        if text_valid is not None:
            interaction_t = interaction_t.masked_fill(~text_valid.unsqueeze(1), -1e4)
        topk_txt = interaction_t.topk(k_txt, dim=-1).indices
        scores_v2t = interaction_t.gather(dim=-1, index=topk_txt)
        att_v2t = F.softmax(scores_v2t, dim=-1)
        if self.training and float(edge_dropout) > 0:
            att_v2t = F.dropout(att_v2t, p=float(edge_dropout), training=True)
            att_v2t = att_v2t / (att_v2t.sum(dim=-1, keepdim=True) + 1e-8)
        text_values = self.text_value(text_states)
        gathered_text_values = text_values[batch_index, topk_txt]
        msg_image = (att_v2t.unsqueeze(-1) * gathered_text_values).sum(dim=-2)

        if self.use_global:
            text_mean = self._masked_mean(text_states, attention_mask)
            image_mean = image_states.mean(dim=1)
            global_input = self.global_from_nodes(torch.cat((text_mean, image_mean), dim=-1))
            if global_state is None:
                global_state = global_input.unsqueeze(1)
            global_state = self.global_ln(global_state + self.global_ffn(global_input).unsqueeze(1))
            msg_text = msg_text + self.global_to_text(global_state).squeeze(1).unsqueeze(1)
            msg_image = msg_image + self.global_to_image(global_state).squeeze(1).unsqueeze(1)

        text_states = self.text_ln(text_states + self.text_ffn(self.msg_dropout(msg_text)))
        image_states = self.image_ln(image_states + self.image_ffn(self.msg_dropout(msg_image)))

        return text_states, image_states, global_state


class BipartiteGraphReasoner(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_layers: int,
        top_k: int,
        edge_dropout: float,
        dropout_rate: float,
        use_global: bool,
    ):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.top_k = int(top_k)
        self.edge_dropout = float(edge_dropout)
        self.use_global = bool(use_global)
        self.layers = nn.ModuleList(
            [
                BipartiteGraphLayer(
                    hidden_size=self.hidden_size,
                    dropout_rate=float(dropout_rate),
                    use_global=self.use_global,
                )
                for _ in range(self.num_layers)
            ]
        )

    def forward(self, text_states, image_states, interaction, attention_mask, edge_dropout=None):
        if edge_dropout is None:
            edge_dropout = self.edge_dropout
        global_state = None
        for layer in self.layers:
            text_states, image_states, global_state = layer(
                text_states=text_states,
                image_states=image_states,
                interaction=interaction,
                attention_mask=attention_mask,
                top_k=self.top_k,
                edge_dropout=float(edge_dropout),
                global_state=global_state,
            )
        if not self.use_global:
            global_state = None
        return text_states, image_states, global_state

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

        self.head_fusion = str(getattr(args, "head_fusion", "add"))
        self.head_weight_delta = float(getattr(args, "head_weight_delta", 1.0))
        self.head_weight_normalize = int(getattr(args, "head_weight_normalize", 1))
        self.head_mul_oracle_lambda = float(getattr(args, "head_mul_oracle_lambda", 0.0))
        self.head_mul_oracle_tau = float(getattr(args, "head_mul_oracle_tau", 0.0))
        self.head_mul_oracle_train_alpha = float(getattr(args, "head_mul_oracle_train_alpha", 0.0))
        if self.head_fusion == "learned":
            self.head_weight_logits = nn.Parameter(torch.zeros(3))

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

        self.gnn_enable = int(getattr(args, "gnn_enable", 0))
        if self.gnn_enable == 1:
            self.gnn_layers = int(getattr(args, "gnn_layers", 2))
            self.gnn_top_k = int(getattr(args, "gnn_top_k", -1))
            if self.gnn_top_k < 1:
                self.gnn_top_k = int(self.fim_top_k)
            self.gnn_edge_dropout = float(getattr(args, "gnn_edge_dropout", 0.1))
            self.gnn_use_global = int(getattr(args, "gnn_use_global", 1)) == 1
            self.gnn_alpha = float(getattr(args, "gnn_alpha", 1.0))
            self.gnn_contrastive_weight = float(getattr(args, "gnn_contrastive_weight", 0.0))
            self.gnn_contrastive_temp = float(getattr(args, "gnn_contrastive_temp", 0.07))
            self.gnn_contrastive_edge_dropout = float(getattr(args, "gnn_contrastive_edge_dropout", -1.0))
            if self.gnn_contrastive_edge_dropout < 0:
                self.gnn_contrastive_edge_dropout = self.gnn_edge_dropout
            self.gnn_gate_init = float(getattr(args, "gnn_gate_init", -2.0))
            self.gnn_gate = nn.Parameter(torch.tensor(self.gnn_gate_init))

            self.gnn_reasoner = BipartiteGraphReasoner(
                hidden_size=args.text_size,
                num_layers=self.gnn_layers,
                top_k=self.gnn_top_k,
                edge_dropout=self.gnn_edge_dropout,
                dropout_rate=args.dropout_rate,
                use_global=self.gnn_use_global,
            )
            self.gnn_out = nn.Linear(args.text_size * 2, args.text_size)
            self.gnn_final_fuse = nn.Linear(args.text_size * 2, args.text_size)

        self.loss_fct = nn.CrossEntropyLoss()
        self.att = nn.Linear(args.text_size, 1, bias=False)

    def _head_weights_multiplicative_from_probs(self, modalities_probs, labels):
        delta = float(self.head_weight_delta)
        num_modalities = int(modalities_probs.shape[1])
        power = delta / (float(num_modalities) - 1.0)
        labels_ = labels.to(torch.long).view(-1, 1, 1)
        p_correct = modalities_probs.gather(dim=-1, index=labels_.expand(-1, num_modalities, 1)).squeeze(-1)

        weights = []
        for i in range(num_modalities):
            others = [j for j in range(num_modalities) if j != i]
            prod = torch.ones_like(p_correct[:, i])
            for j in others:
                prod = prod * (1.0 - p_correct[:, j])
            weights.append(prod.clamp(min=1e-6).pow(power))
        weights = torch.stack(weights, dim=1)
        if int(self.head_weight_normalize) == 1:
            weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)
        return weights

    def _head_weights_multiplicative_from_conf(self, conf):
        delta = float(self.head_weight_delta)
        num_modalities = int(conf.shape[1])
        power = delta / (float(num_modalities) - 1.0)

        weights = []
        for i in range(num_modalities):
            others = [j for j in range(num_modalities) if j != i]
            prod = torch.ones_like(conf[:, i])
            for j in others:
                prod = prod * (1.0 - conf[:, j])
            weights.append(prod.clamp(min=1e-6).pow(power))
        weights = torch.stack(weights, dim=1)
        if int(self.head_weight_normalize) == 1:
            weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)
        return weights

    def _head_weights_multiplicative_per_class(self, modalities_probs):
        delta = float(self.head_weight_delta)
        num_modalities = int(modalities_probs.shape[1])
        power = delta / (float(num_modalities) - 1.0)

        one_minus = (1.0 - modalities_probs).clamp(min=1e-6)
        prod_all = one_minus.prod(dim=1, keepdim=True)
        weights = []
        for i in range(num_modalities):
            denom = one_minus[:, i:i + 1, :]
            w_i = (prod_all / denom).clamp(min=1e-6).pow(power)
            weights.append(w_i)
        weights = torch.cat(weights, dim=1)
        if int(self.head_weight_normalize) == 1:
            weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)
        return weights

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

        gnn_contrastive_loss = None
        if self.gnn_enable == 1:
            gnn_text_states, gnn_image_states, gnn_global_state = self.gnn_reasoner(
                text_states=text_embeds,
                image_states=image_embeds,
                interaction=interaction,
                attention_mask=inputs.get('attention_mask', None),
                edge_dropout=self.gnn_edge_dropout,
            )
            gnn_text_feature = gnn_text_states[
                torch.arange(gnn_text_states.shape[0], device=gnn_text_states.device),
                last_token_index,
            ]
            gnn_image_feature = gnn_image_states[:, 0, :].squeeze(1)
            gnn_text_weight = self.att(gnn_text_feature)
            gnn_image_weight = self.att(gnn_image_feature)
            gnn_att = nn.functional.softmax(torch.stack((gnn_text_weight, gnn_image_weight), dim=-1), dim=-1)
            gnn_tw, gnn_iw = gnn_att.split([1, 1], dim=-1)
            gnn_local_fuse = gnn_tw.squeeze(1) * gnn_text_feature + gnn_iw.squeeze(1) * gnn_image_feature
            if gnn_global_state is None:
                gnn_feature = gnn_local_fuse
            else:
                gnn_feature = self.gnn_out(torch.cat((gnn_local_fuse, gnn_global_state.squeeze(1)), dim=-1))
            gnn_gate = torch.sigmoid(self.gnn_gate) * float(self.gnn_alpha)
            gnn_feature = gnn_feature * gnn_gate
            fuse_feature = self.gnn_final_fuse(torch.cat((fuse_feature, gnn_feature), dim=-1))

            if (
                labels is not None
                and self.training
                and float(getattr(self, "gnn_contrastive_weight", 0.0)) > 0
            ):
                g2_text_states, g2_image_states, g2_global_state = self.gnn_reasoner(
                    text_states=text_embeds,
                    image_states=image_embeds,
                    interaction=interaction,
                    attention_mask=inputs.get('attention_mask', None),
                    edge_dropout=self.gnn_contrastive_edge_dropout,
                )
                g2_text_feature = g2_text_states[
                    torch.arange(g2_text_states.shape[0], device=g2_text_states.device),
                    last_token_index,
                ]
                g2_image_feature = g2_image_states[:, 0, :].squeeze(1)
                g2_text_weight = self.att(g2_text_feature)
                g2_image_weight = self.att(g2_image_feature)
                g2_att = nn.functional.softmax(
                    torch.stack((g2_text_weight, g2_image_weight), dim=-1), dim=-1
                )
                g2_tw, g2_iw = g2_att.split([1, 1], dim=-1)
                g2_local_fuse = g2_tw.squeeze(1) * g2_text_feature + g2_iw.squeeze(1) * g2_image_feature
                if g2_global_state is None:
                    gnn_feature_2 = g2_local_fuse
                else:
                    gnn_feature_2 = self.gnn_out(
                        torch.cat((g2_local_fuse, g2_global_state.squeeze(1)), dim=-1)
                    )

                z1 = F.normalize(gnn_feature, p=2, dim=-1)
                z2 = F.normalize(gnn_feature_2, p=2, dim=-1)
                logits = torch.matmul(z1, z2.transpose(0, 1)) / max(float(self.gnn_contrastive_temp), 1e-6)
                contrastive_targets = torch.arange(logits.shape[0], device=logits.device)
                gnn_contrastive_loss = 0.5 * (
                    self.loss_fct(logits, contrastive_targets) + self.loss_fct(logits.transpose(0, 1), contrastive_targets)
                )

        # 通过三个分类头得到未归一化的分类分数
        logits_fuse = self.classifier_fuse(fuse_feature)
        logits_text = self.classifier_text(text_feature)
        logits_image = self.classifier_image(image_feature)
   
        # 用 softmax 把 logits 变成每个类别的概率
        fuse_score = nn.functional.softmax(logits_fuse, dim=-1)
        text_score = nn.functional.softmax(logits_text, dim=-1)
        image_score = nn.functional.softmax(logits_image, dim=-1)

        modalities = torch.stack([fuse_score, text_score, image_score], dim=1)

        if self.head_fusion == "add":
            score = fuse_score + text_score + image_score
        elif self.head_fusion == "learned":
            weights = F.softmax(self.head_weight_logits, dim=-1)
            score = weights[0] * fuse_score + weights[1] * text_score + weights[2] * image_score
        elif self.head_fusion == "mul":
            weights_pc = self._head_weights_multiplicative_per_class(modalities)
            score = (weights_pc * modalities).sum(dim=1)

            if (
                (not self.training)
                and labels is not None
                and float(self.head_mul_oracle_lambda) > 0
                and float(self.head_mul_oracle_tau) > 0
            ):
                top2 = score.topk(k=2, dim=-1).values
                margin = top2[:, 0] - top2[:, 1]
                pred = score.argmax(dim=-1)
                use_oracle = (margin < float(self.head_mul_oracle_tau)) & (pred != labels.to(torch.long))
                if use_oracle.any():
                    oracle_weights = self._head_weights_multiplicative_from_probs(modalities, labels)
                    oracle_score = (oracle_weights.unsqueeze(-1) * modalities).sum(dim=1)
                    lam = float(self.head_mul_oracle_lambda)
                    tau = max(float(self.head_mul_oracle_tau), 1e-8)
                    lam_vec = (lam * ((tau - margin) / tau).clamp(min=0.0, max=1.0)).to(score.dtype)
                    score = score.clone()
                    score[use_oracle] = (
                        (1.0 - lam_vec[use_oracle].unsqueeze(-1)) * score[use_oracle]
                        + lam_vec[use_oracle].unsqueeze(-1) * oracle_score[use_oracle]
                    )
        else:
            score = fuse_score + text_score + image_score

        outputs = (score,)
        if labels is not None:
            if self.head_fusion == "mul":
                logits_all = torch.stack([logits_fuse, logits_text, logits_image], dim=1)
                logp = F.log_softmax(logits_all, dim=-1)
                labels_ = labels.to(torch.long).view(-1, 1, 1)
                logp_y = logp.gather(dim=-1, index=labels_.expand(-1, 3, 1)).squeeze(-1)
                nll = -logp_y
                weights = self._head_weights_multiplicative_from_probs(modalities, labels)
                loss = (weights * nll).sum(dim=1).mean()

                alpha = float(getattr(self, "head_mul_oracle_train_alpha", 0.0))
                if alpha > 0:
                    oracle_score = (weights.unsqueeze(-1) * modalities).sum(dim=1)
                    eps = 1e-8
                    p = score.clamp(min=eps)
                    p = p / p.sum(dim=-1, keepdim=True).clamp(min=eps)
                    q = oracle_score.clamp(min=eps)
                    q = q / q.sum(dim=-1, keepdim=True).clamp(min=eps)
                    distill_loss = (q * (q.log() - p.log())).sum(dim=-1).mean()
                    loss = loss + alpha * distill_loss
            elif self.head_fusion == "learned":
                weights = F.softmax(self.head_weight_logits, dim=-1)
                loss_fuse = self.loss_fct(logits_fuse, labels)
                loss_text = self.loss_fct(logits_text, labels)
                loss_image = self.loss_fct(logits_image, labels)
                loss = weights[0] * loss_fuse + weights[1] * loss_text + weights[2] * loss_image
            else:
                loss_fuse = self.loss_fct(logits_fuse, labels)
                loss_text = self.loss_fct(logits_text, labels)
                loss_image = self.loss_fct(logits_image, labels)
                loss = loss_fuse + loss_text + loss_image

            if gnn_contrastive_loss is not None:
                loss = loss + float(self.gnn_contrastive_weight) * gnn_contrastive_loss

            outputs = (loss,) + outputs
        return outputs


class RoBERTaViTFusion(nn.Module):
    def __init__(self, args):
        super().__init__()
        text_encoder_name = str(getattr(args, "text_encoder_name", "roberta-base"))
        vision_encoder_name = str(getattr(args, "vision_encoder_name", "google/vit-base-patch16-224"))
        cache_dir = getattr(args, "hf_cache_dir", None)
        self.text_encoder = RobertaModel.from_pretrained(text_encoder_name, cache_dir=cache_dir)
        self.vision_encoder = ViTModel.from_pretrained(vision_encoder_name, cache_dir=cache_dir)

        text_dim = int(self.text_encoder.config.hidden_size)
        image_dim = int(self.vision_encoder.config.hidden_size)
        fusion_dim = int(getattr(args, "fusion_dim", 512))
        dropout = float(getattr(args, "dropout_rate", 0.1))

        self.fusion = nn.Sequential(
            nn.Linear(text_dim + image_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(fusion_dim // 2, int(getattr(args, "label_number", 2)))
        self.loss_fct = nn.CrossEntropyLoss()

    def forward(self, inputs, labels):
        text_out = self.text_encoder(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask", None),
        )
        text_feature = text_out.last_hidden_state[:, 0, :]

        vis_out = self.vision_encoder(pixel_values=inputs["pixel_values"])
        if getattr(vis_out, "pooler_output", None) is None:
            image_feature = vis_out.last_hidden_state[:, 0, :]
        else:
            image_feature = vis_out.pooler_output

        fused = self.fusion(torch.cat([text_feature, image_feature], dim=-1))
        logits = self.classifier(fused)
        score = F.softmax(logits, dim=-1)
        outputs = (score,)
        if labels is not None:
            loss = self.loss_fct(logits, labels)
            outputs = (loss,) + outputs
        return outputs
