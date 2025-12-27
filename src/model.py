import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformers import AutoConfig, CLIPProcessor, CLIPModel, CLIPTextModel, CLIPVisionModel

class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        # 为query单独创建投影层
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        # 为context的key和value创建投影层
        self.kv_proj = nn.Linear(dim, dim * 2, bias=qkv_bias)
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, context=None):
        B, N, C = x.shape
        if context is None:
            context = x
        
        # 确保context的通道数与x相同
        if context.shape[-1] != C:
            # 如果维度不同，进行投影
            proj = nn.Linear(context.shape[-1], C).to(context.device)
            context = proj(context)
        
        # 计算查询
        q = self.q_proj(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        
        # 使用context计算键和值
        kv = self.kv_proj(context).reshape(B, context.shape[1], 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class MultimodalEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.dim = args.text_size + args.image_size
        
        # 加载预训练的CLIP模型
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        
        # 交叉注意力层
        self.text_attention = CrossAttention(args.text_size, num_heads=args.heads)
        self.image_attention = CrossAttention(args.image_size, num_heads=args.heads)
        
        # 投影层
        self.text_proj = nn.Linear(args.text_size, args.text_size)
        self.image_proj = nn.Linear(args.image_size, args.image_size)
        
        # 融合层
        self.fusion_proj = nn.Linear(self.dim, args.fusion_dim)
        self.dropout = nn.Dropout(args.dropout_rate)
        
        # 分类头
        self.classifier = nn.Linear(args.fusion_dim, args.label_number)
        
    def forward(self, input_ids=None, attention_mask=None, pixel_values=None, labels=None):
        # 获取文本特征
        text_outputs = self.clip_model.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        text_features = text_outputs.last_hidden_state[:, 0]
        
        # 获取视觉特征
        image_outputs = self.clip_model.vision_model(
            pixel_values=pixel_values,
            output_hidden_states=True
        )
        image_features = image_outputs.last_hidden_state[:, 0]
        
        # 残差连接的交叉注意力
        # 注意：当将image_features传给text_attention时，需要确保维度匹配
        # 同样，当将text_features传给image_attention时，也需要确保维度匹配
        text_features_attn = self.text_attention(
            text_features.unsqueeze(1), 
            image_features.unsqueeze(1)
        ).squeeze(1)
        text_features = F.gelu(self.text_proj(text_features)) + text_features_attn
        
        image_features_attn = self.image_attention(
            image_features.unsqueeze(1), 
            text_features.unsqueeze(1)
        ).squeeze(1)
        image_features = F.gelu(self.image_proj(image_features)) + image_features_attn
        
        # 融合特征
        combined_features = torch.cat([text_features, image_features], dim=-1)
        fusion_features = F.gelu(self.fusion_proj(combined_features))
        fusion_features = self.dropout(fusion_features)
        
        # 分类
        logits = self.classifier(fusion_features)
        
        # 计算对比损失
        if hasattr(self.args, 'use_contrastive_loss') and self.args.use_contrastive_loss:
            # 确保文本特征和图像特征具有相同的维度
            if text_features.shape[-1] != image_features.shape[-1]:
                # 创建投影层使维度匹配
                if text_features.shape[-1] > image_features.shape[-1]:
                    proj = nn.Linear(image_features.shape[-1], text_features.shape[-1]).to(image_features.device)
                    image_features_proj = proj(image_features)
                    text_embeds = F.normalize(text_features, dim=-1)
                    image_embeds = F.normalize(image_features_proj, dim=-1)
                else:
                    proj = nn.Linear(text_features.shape[-1], image_features.shape[-1]).to(text_features.device)
                    text_features_proj = proj(text_features)
                    text_embeds = F.normalize(text_features_proj, dim=-1)
                    image_embeds = F.normalize(image_features, dim=-1)
            else:
                text_embeds = F.normalize(text_features, dim=-1)
                image_embeds = F.normalize(image_features, dim=-1)
            
            # 计算文本和图像之间的相似度矩阵
            logits_per_text = torch.matmul(text_embeds, image_embeds.t()) * math.exp(self.args.temperature)
            logits_per_image = logits_per_text.t()
            
            # 创建标签（对角线）
            batch_size = text_embeds.shape[0]
            labels_contrastive = torch.arange(batch_size, device=text_embeds.device)
            
            # 计算对比损失
            contrastive_loss = (
                F.cross_entropy(logits_per_text, labels_contrastive) +
                F.cross_entropy(logits_per_image, labels_contrastive)
            ) / 2
        else:
            contrastive_loss = 0.0
        
        # 计算分类损失
        if labels is not None:
            # 使用Focal Loss
            if hasattr(self.args, 'use_focal_loss') and self.args.use_focal_loss:
                ce_loss = F.cross_entropy(logits, labels, reduction='none')
                pt = torch.exp(-ce_loss)
                class_loss = (1 - pt) ** self.args.focal_gamma * ce_loss
                class_loss = class_loss.mean()
            else:
                class_loss = F.cross_entropy(logits, labels)
            
            # 组合损失
            contrastive_weight = getattr(self.args, 'contrastive_weight', 0.1)
            loss = class_loss + contrastive_weight * contrastive_loss
            return loss, logits
        
        return logits

class MV_CLIP(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.model = MultimodalEncoder(args)
        
    def forward(self, input_ids=None, attention_mask=None, pixel_values=None, labels=None):
        # 直接传递所有参数给内部模型
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            labels=labels
        )