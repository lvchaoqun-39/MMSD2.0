from transformers import CLIPModel,BertConfig,BertModel
from transformers.models.bert.modeling_bert import BertLayer
import torch.nn as nn
import torch
import torch.nn.functional as F
import copy
import math

class MultimodalEncoder(nn.Module):
    def __init__(self, config, layer_number):
        super(MultimodalEncoder, self).__init__()
        layer = BertLayer(config)
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
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", local_files_only=True)
        self.config = BertConfig.from_pretrained("bert-base-uncased", local_files_only=True)
        self.config.hidden_size = 512
        self.config.num_attention_heads = 8
        self.trans = MultimodalEncoder(self.config, layer_number=args.layers)
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

        self.loss_fct = nn.CrossEntropyLoss()
        self.att = nn.Linear(args.text_size, 1, bias=False)

    def forward(self, inputs, labels=None):
        output = self.model(**inputs,output_attentions=True)
        text_features = output['text_model_output']['last_hidden_state']
        image_features = output['vision_model_output']['last_hidden_state']
        text_feature = output['text_model_output']['pooler_output']
        image_feature = output['vision_model_output']['pooler_output']
        text_feature = self.text_linear(text_feature)
        image_feature = self.image_linear(image_feature)

        text_embeds = self.model.text_projection(text_features)
        image_embeds = self.model.visual_projection(image_features)
        input_embeds = torch.cat((image_embeds, text_embeds), dim=1)
        attention_mask = torch.cat((torch.ones(text_features.shape[0], 50).to(text_features.device), inputs['attention_mask']), dim=-1)
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        fuse_hiddens, all_attentions = self.trans(input_embeds, extended_attention_mask, output_all_encoded_layers=False)
        fuse_hiddens = fuse_hiddens[-1]
        new_text_features = fuse_hiddens[:, 50:, :]
        new_text_feature = new_text_features[
            torch.arange(new_text_features.shape[0], device=inputs['input_ids'].device), inputs['input_ids'].to(torch.int).argmax(dim=-1)
        ]

        new_image_feature = fuse_hiddens[:, 0, :].squeeze(1)

        text_weight = self.att(new_text_feature)
        image_weight = self.att(new_image_feature)    
        att = nn.functional.softmax(torch.stack((text_weight, image_weight), dim=-1),dim=-1)
        tw, iw = att.split([1,1], dim=-1)
        fuse_feature = tw.squeeze(1) * new_text_feature + iw.squeeze(1) * new_image_feature

        logits_fuse = self.classifier_fuse(fuse_feature)
        logits_text = self.classifier_text(text_feature)
        logits_image = self.classifier_image(image_feature)
   
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


class RelationGraphAttention(nn.Module):
    """
    关系图注意力模块，用于建模模态内和模态间的关系
    """
    def __init__(self, hidden_size, num_heads=4):
        super(RelationGraphAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        # 关系建模的线性变换
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        
        # 关系类型嵌入
        self.rel_embedding = nn.Embedding(3, hidden_size)  # 0: 同一模态内部关系, 1: 文本->图像关系, 2: 图像->文本关系
    
    def forward(self, text_features, image_features):
        batch_size = text_features.size(0)
        
        # 合并特征
        all_features = torch.cat([text_features, image_features], dim=1)
        seq_len = all_features.size(1)
        
        # 生成关系掩码
        text_len = text_features.size(1)
        img_len = image_features.size(1)
        
        # 创建关系类型矩阵
        rel_type = torch.zeros((batch_size, seq_len, seq_len), device=all_features.device, dtype=torch.long)
        # 文本内部关系
        rel_type[:, :text_len, :text_len] = 0
        # 图像内部关系
        rel_type[:, text_len:, text_len:] = 0
        # 文本到图像的关系
        rel_type[:, :text_len, text_len:] = 1
        # 图像到文本的关系
        rel_type[:, text_len:, :text_len] = 2
        
        # 获取关系嵌入
        rel_embeddings = self.rel_embedding(rel_type)
        
        # 计算注意力
        q = self.query(all_features).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(all_features).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(all_features).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 重新设计关系嵌入的使用方式，避免维度不匹配
        # 先进行标准的注意力计算
        attn = torch.matmul(q, k.transpose(-2, -1))
        attn = attn / math.sqrt(self.head_dim)
        
        # 关系嵌入作为额外的注意力偏置
        rel_emb = rel_embeddings.mean(dim=-1)  # 降维处理
        attn = attn + rel_emb.view(batch_size, 1, seq_len, seq_len)
        
        attn = F.softmax(attn, dim=-1)
        
        # 加权聚合
        context = torch.matmul(attn, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        
        # 输出投影
        output = self.out_proj(context)
        
        return output


class MultiplexFusion(nn.Module):
    """
    多路融合模块，用于多层次特征融合
    """
    def __init__(self, hidden_size, fusion_layers=2):
        super(MultiplexFusion, self).__init__()
        self.hidden_size = hidden_size
        self.fusion_layers = fusion_layers
        
        # 多层次融合
        self.fusion_blocks = nn.ModuleList()
        for _ in range(fusion_layers):
            self.fusion_blocks.append(nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.GELU(),
                nn.Dropout(0.1)
            ))
        
        # 融合权重计算
        self.weight_layer = nn.Linear(hidden_size * 3, 3)
    
    def forward(self, text_feature, image_feature, cross_feature):
        # 初始特征 - 直接使用cross_feature而不是拼接
        curr_fusion = cross_feature
        
        # 多层次融合
        for block in self.fusion_blocks:
            curr_fusion = block(curr_fusion)
        
        # 计算动态融合权重
        weight_input = torch.cat([text_feature, image_feature, cross_feature], dim=-1)
        weights = F.softmax(self.weight_layer(weight_input), dim=-1)
        
        # 加权融合
        final_feature = weights[:, 0:1] * text_feature + \
                        weights[:, 1:2] * image_feature + \
                        weights[:, 2:3] * curr_fusion
        
        return final_feature


class RCLMFN(nn.Module):
    """
    Relational Context Learning and Multiplex Fusion Network
    关系上下文学习和多路融合网络
    """
    def __init__(self, args):
        super(RCLMFN, self).__init__()
        # 使用BERT处理文本，CLIP的视觉模型处理图像
        self.text_encoder = BertModel.from_pretrained("bert-base-uncased", local_files_only=True)
        self.image_encoder = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", local_files_only=True).vision_model
        
        # 特征维度调整
        self.text_proj = nn.Linear(768, args.hidden_size)
        self.image_proj = nn.Linear(768, args.hidden_size)
        
        # 关系建模模块
        self.relational_encoder = RelationGraphAttention(args.hidden_size, num_heads=args.num_heads)
        
        # 多路融合模块
        self.multiplex_fusion = MultiplexFusion(args.hidden_size, fusion_layers=args.fusion_layers)
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(args.hidden_size, args.hidden_size // 2),
            nn.GELU(),
            nn.Dropout(args.dropout_rate),
            nn.Linear(args.hidden_size // 2, args.label_number)
        )
        
        # 损失函数
        self.loss_fct = nn.CrossEntropyLoss()
    
    def forward(self, inputs, labels=None):
        # 适配不同的输入格式，兼容data_set.py中的处理逻辑
        if isinstance(inputs, dict) and 'text' in inputs and 'image' in inputs:
            # 从字典中获取text和image输入
            text_inputs = inputs['text']
            image_inputs = inputs['image']
        else:
            # 直接使用输入（兼容旧格式）
            text_inputs = inputs
            image_inputs = inputs
        
        # 文本特征提取
        text_output = self.text_encoder(
            input_ids=text_inputs.get('input_ids'),
            attention_mask=text_inputs.get('attention_mask')
        )
        text_features = text_output.last_hidden_state  # [batch, seq_len, 768]
        text_feature = text_output.pooler_output  # [batch, 768]
        
        # 图像特征提取
        image_output = self.image_encoder(image_inputs.get('pixel_values'))
        image_features = image_output.last_hidden_state  # [batch, img_seq_len, 768]
        image_feature = image_output.pooler_output  # [batch, 768]
        
        # 特征维度调整
        text_features = self.text_proj(text_features)  # [batch, seq_len, hidden_size]
        image_features = self.image_proj(image_features)  # [batch, img_seq_len, hidden_size]
        text_feature = self.text_proj(text_feature)  # [batch, hidden_size]
        image_feature = self.image_proj(image_feature)  # [batch, hidden_size]
        
        # 关系建模
        relational_features = self.relational_encoder(text_features, image_features)
        
        # 提取跨模态特征表示
        cross_feature = relational_features.mean(dim=1)  # [batch, hidden_size]
        
        # 多路融合
        fused_feature = self.multiplex_fusion(text_feature, image_feature, cross_feature)
        
        # 分类
        logits = self.classifier(fused_feature)
        scores = F.softmax(logits, dim=-1)
        
        outputs = (scores,)
        if labels is not None:
            loss = self.loss_fct(logits, labels)
            outputs = (loss,) + outputs
        
        return outputs