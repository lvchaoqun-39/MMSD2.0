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
        
        # 获取CLIP模型的实际输出维度
        self.actual_text_size = self.clip_model.text_model.config.hidden_size
        self.actual_image_size = self.clip_model.vision_model.config.hidden_size
        
        # 交叉注意力层 - 使用实际维度
        self.text_attention = CrossAttention(self.actual_text_size, num_heads=args.heads)
        self.image_attention = CrossAttention(self.actual_image_size, num_heads=args.heads)
        
        # 投影层
        self.text_proj = nn.Linear(self.actual_text_size, self.actual_text_size)
        self.image_proj = nn.Linear(self.actual_image_size, self.actual_image_size)
        
        # 融合层
        self.fusion_proj = nn.Linear(self.actual_text_size + self.actual_image_size, args.fusion_dim)
        self.dropout = nn.Dropout(args.dropout_rate)
        
        # 分类头
        self.classifier = nn.Linear(args.fusion_dim, args.label_number)
        
    def forward(self, input_ids, attention_mask, pixel_values, labels=None):
        # 获取文本特征
        text_outputs = self.clip_model.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        text_features = text_outputs.pooler_output  # 形状应该是 [batch_size, actual_text_size]
        
        # 获取图像特征
        image_outputs = self.clip_model.vision_model(
            pixel_values=pixel_values
        )
        image_features = image_outputs.pooler_output  # 形状应该是 [batch_size, actual_image_size]
        
        # 使用交叉注意力 - 根据CrossAttention接口，只需要提供查询和上下文
        text_attended = self.text_attention(
            text_features.unsqueeze(1),
            image_features.unsqueeze(1)
        ).squeeze(1)
        
        image_attended = self.image_attention(
            image_features.unsqueeze(1),
            text_features.unsqueeze(1)
        ).squeeze(1)
        
        # 投影和融合
        text_proj = self.text_proj(text_attended)
        image_proj = self.image_proj(image_attended)
        
        # 合并特征
        combined = torch.cat([text_proj, image_proj], dim=1)
        
        # 通过融合层
        fused = self.fusion_proj(combined)
        fused = self.dropout(fused)
        fused = F.relu(fused)
        
        # 分类
        logits = self.classifier(fused)
        
        # 计算损失
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
        
        return loss, logits

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