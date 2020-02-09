import torch
import torch.nn as nn

from transformers import BertModel

class ModelBuilder(nn.Module):
    def __init__(self, bert_):
        super(ModelBuilder, self).__init__()

        self.bert = BertModel.from_pretrained(bert_)

    def forward(self, input_ids):
        text_embeds = self.bert(input_ids)