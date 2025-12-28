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

        self.loss_fct = nn.CrossEntropyLoss()
        self.att = nn.Linear(args.text_size, 1, bias=False)

    def forward(self, inputs, labels):
        output = self.model(**inputs,output_attentions=True)
        text_features = output['text_model_output']['last_hidden_state'] # 文本特征
        image_features = output['vision_model_output']['last_hidden_state'] # 图像特征
        text_feature = output['text_model_output']['pooler_output'] # 文本池化特征
        image_feature = output['vision_model_output']['pooler_output'] # 图像池化特征
        text_feature = self.text_linear(text_feature) # 文本特征线性变换
        image_feature = self.image_linear(image_feature) # 图像特征线性变换

        text_embeds = self.model.text_projection(text_features) # 文本特征投影
        image_embeds = self.model.visual_projection(image_features) # 图像特征投影
        input_embeds = torch.cat((image_embeds, text_embeds), dim=1) # 合并文本特征和图像特征成为多模态嵌入特征
        attention_mask = torch.cat((torch.ones(text_features.shape[0], 50).to(text_features.device), inputs['attention_mask']), dim=-1) # 合并文本特征和图像特征的注意力掩码
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2) # 扩展掩码维度，适配Transformer注意力计算
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # 将注意力掩码转换为与模型参数 相同的数据类型
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0 # 将注意力掩码转换为 注意力分数屏蔽值
        fuse_hiddens, all_attentions = self.trans(input_embeds, extended_attention_mask, output_all_encoded_layers=False) # 多模态特征融合的核心步骤 ，通过自定义的Transformer编码器将图像和文本特征进行深度交互融合，输出融合后的隐藏状态和所有层的注意力矩阵。
        fuse_hiddens = fuse_hiddens[-1] # 取 Transformer 编码器的最后一层输出作为融合后的隐藏状态
        new_text_features = fuse_hiddens[:, 50:, :] # 提取文本部分的融合特征，从第 50 个 token 开始到结束
        new_text_feature = new_text_features[
            torch.arange(new_text_features.shape[0], device=inputs['input_ids'].device), inputs['input_ids'].to(torch.int).argmax(dim=-1)
        ] # 提取文本序列的全局特征，从文本序列特征中选择 最后一个有效token的特征 作为整个文本的融合表示

        new_image_feature = fuse_hiddens[:, 0, :].squeeze(1) # 提取图像部分的融合特征，取第 0 个 token 的特征（CLS 令牌）

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


