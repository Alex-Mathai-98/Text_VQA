import torch
import torch.nn as nn

from transformers import BertModel

class TextModel(nn.Module):
    def __init__(self, checkpoint):
        super(TextModel, self).__init__()

        self.bert = BertModel.from_pretrained(checkpoint)

    def forward(self, input_ids, attention_masks):
        text_embeds = self.bert(input_ids)
        return text_embeds, text_embeds[:, 0, :]