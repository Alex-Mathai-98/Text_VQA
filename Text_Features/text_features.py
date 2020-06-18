import torch
import torch.nn as nn
import pickle
import os
from transformers import BertModel
from transformers import BertTokenizer
from typing import List

class BertTextModel(nn.Module):
	def __init__(self, checkpoint):
		super(BertTextModel, self).__init__()

		self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
		dir_name = os.path.dirname(__file__)

		with open( os.path.join(dir_name,'oov_tokens.pkl'), 'rb' ) as f:
			SPECIAL_TOKENS = pickle.load(f)
		self.tokenizer.add_tokens(SPECIAL_TOKENS)

		self.bert = BertModel.from_pretrained(checkpoint)
		new_size = self.bert.resize_token_embeddings(len(self.tokenizer))

	def forward(self, sent_list:List[str]):

		maxlen = -1
		lengths = []
		input_list = []
		for sent in sent_list :
			words = self.tokenizer.tokenize(sent)
			input_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(sent))
			input_ids = self.tokenizer.build_inputs_with_special_tokens(input_ids)
			input_list.append(input_ids)
			lengths.append(len(input_ids))
			maxlen = max(maxlen,lengths[-1])
			#temp = self.tokenizer.convert_ids_to_tokens(input_ids)
			#print(temp)
		lengths = torch.tensor(lengths)

		for idx,length in enumerate(lengths) :
			if length < maxlen :

				rem_list = []
				for m in range(maxlen-length):
					rem_list.append(self.tokenizer.pad_token_id)

				input_list[idx] = input_list[idx] + rem_list

		mask = torch.arange(maxlen)[None, :] < lengths[:, None]

		input_ids = torch.tensor(input_list)

		all_steps,final = self.bert(input_ids,mask)

		return all_steps,final

if __name__ == '__main__' :

	sent = ["I love dancing", "You hate me", "No I wont"]
	alex = BertTextModel("/home/alex/Desktop/4-2/Text_VQA/utils/output/")

	all_steps,final = alex.forward(sent)
	print(final.size())