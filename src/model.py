from transformers import CLIPModel, BertConfig, BertModel
from transformers.models.bert.modeling_bert import BertLayer
import torch.nn as nn
import torch
import torch.nn.functional as F
import copy
from torchvision import models

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
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.config = BertConfig.from_pretrained("bert-base-uncased")
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

    def forward(self, inputs, labels):
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


class ResNetImageEncoder(nn.Module):
    def __init__(self, model_name="resnet50"):
        super().__init__()
        if model_name != "resnet50":
            raise ValueError("Only resnet50 is supported")
        try:
            weights = models.ResNet50_Weights.DEFAULT
            resnet = models.resnet50(weights=weights)
        except Exception:
            resnet = models.resnet50(pretrained=True)

        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.output_dim = 2048

    def forward(self, images):
        feature_map = self.backbone(images)
        pooled = self.avgpool(feature_map).flatten(1)
        tokens = feature_map.flatten(2).transpose(1, 2)
        return tokens, pooled


class MV_BERT_RESNET(nn.Module):
    def __init__(self, args):
        super(MV_BERT_RESNET, self).__init__()
        self.text_encoder = BertModel.from_pretrained("bert-base-uncased")
        self.image_encoder = ResNetImageEncoder(model_name="resnet50")

        hidden_size = self.text_encoder.config.hidden_size
        self.hidden_size = hidden_size

        self.config = BertConfig.from_pretrained("bert-base-uncased")
        self.config.hidden_size = hidden_size
        self.config.num_attention_heads = max(1, hidden_size // 64)
        self.trans = MultimodalEncoder(self.config, layer_number=args.layers)

        if args.simple_linear:
            self.text_linear = nn.Linear(hidden_size, hidden_size)
            self.image_linear = nn.Linear(hidden_size, hidden_size)
        else:
            self.text_linear = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.Dropout(args.dropout_rate),
                nn.GELU(),
            )
            self.image_linear = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.Dropout(args.dropout_rate),
                nn.GELU(),
            )

        self.image_token_projection = nn.Linear(self.image_encoder.output_dim, hidden_size)
        self.image_pool_projection = nn.Linear(self.image_encoder.output_dim, hidden_size)

        self.classifier_fuse = nn.Linear(hidden_size, args.label_number)
        self.classifier_text = nn.Linear(hidden_size, args.label_number)
        self.classifier_image = nn.Linear(hidden_size, args.label_number)

        self.loss_fct = nn.CrossEntropyLoss()
        self.att = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, inputs, labels):
        text_outputs = self.text_encoder(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask", None),
            token_type_ids=inputs.get("token_type_ids", None),
            output_attentions=True,
            return_dict=True,
        )

        text_features = text_outputs.last_hidden_state
        text_feature = text_outputs.pooler_output
        text_feature = self.text_linear(text_feature)

        image_tokens_raw, image_pooled_raw = self.image_encoder(inputs["images"])
        image_embeds = self.image_token_projection(image_tokens_raw)
        image_feature = self.image_pool_projection(image_pooled_raw)
        image_feature = self.image_linear(image_feature)

        input_embeds = torch.cat((image_embeds, text_features), dim=1)
        image_attention = torch.ones(
            text_features.shape[0],
            image_embeds.shape[1],
            device=text_features.device,
            dtype=inputs["attention_mask"].dtype if "attention_mask" in inputs else torch.long,
        )
        if "attention_mask" in inputs:
            attention_mask = torch.cat((image_attention, inputs["attention_mask"]), dim=-1)
        else:
            attention_mask = image_attention
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        fuse_hiddens, _ = self.trans(input_embeds, extended_attention_mask, output_all_encoded_layers=False)
        fuse_hiddens = fuse_hiddens[-1]
        new_image_feature = fuse_hiddens[:, 0, :]
        new_text_feature = fuse_hiddens[:, image_embeds.shape[1], :]

        text_weight = self.att(new_text_feature)
        image_weight = self.att(new_image_feature)
        att = nn.functional.softmax(torch.stack((text_weight, image_weight), dim=-1), dim=-1)
        tw, iw = att.split([1, 1], dim=-1)
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

