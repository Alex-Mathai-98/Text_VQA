import torch
from torch.utils import data
from torchvision import transforms
from PIL import Image
import json
import os
import numpy as np
import tqdm
from transformers import BertTokenizer, BertModel

class CustomDataset(data.Dataset):
	'Characterizes a dataset for PyTorch'
	def __init__(self, data_path, ID_path,json_path,target_image_size,set_="train"):
		'Initialization'
		self.data_path = data_path
		self.cleaned_json = self.read_Json(json_path)
		self.list_IDs = self.read_IDs(ID_path)
		self.target_image_size = target_image_size
		self.set_ = set_
		self.channel_mean = [0.485, 0.456, 0.406]
		self.channel_std = [0.229, 0.224, 0.225]

		self.color_transforms = transforms.Compose([
				transforms.Resize(self.target_image_size),
				transforms.ToTensor(),
				transforms.Normalize(self.channel_mean, self.channel_std)])

	def read_IDs(self,file) :
		ans = []
		with open(file,"r") as f:
			for line in f:
				ans.append(line[:-1])
		return ans

	def read_Json(self,file) :
		with open(file) as f:
			json_file = json.load(f)
		return json_file

	def __len__(self):
		'Denotes the total number of samples'
		return len(self.list_IDs)
	
	def get_path(self, index):
		ID = self.list_IDs[index]
		return str(os.path.join(self.data_path, self.set_+"/" + self.set_ + "_images/"+ID+".jpg"))

	def __getitem__(self, index):
		'Generates one sample of data'

		# Select sample
		ID = self.cleaned_json["image_ids"][self.list_IDs[index]]
		# print (ID)
		# Load image
		X = Image.open(os.path.join(data_path,self.set_+"/" + self.set_ + "_images/"+ID+".jpg"))
		
		if X.mode != 'RGB' :
			X = X.convert('RGB')

		X = self.color_transforms(X)

		# Load question
		q = self.cleaned_json["question"][self.list_IDs[index]]

		if self.set_ == "test" :
			return X,q
		else :
			# Load Label
			y = self.cleaned_json["answers"][self.list_IDs[index]][0]
			return X,q,y


if __name__ == '__main__' :

	data_path = "/home/alex/Desktop/4-2/Text_VQA/Data/"
	ID_path = os.path.join(data_path,"train/train_ids.txt")
	json_path = os.path.join(data_path,"train/cleaned.json")
	alex = CustomDataset(data_path,ID_path,json_path,(448,448),set_="train")
	tks = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
	bert_model = BertModel.from_pretrained('bert-base-uncased')
	total = 0
	issue = 0
			
	_alex = data.DataLoader(alex)

	# bert_model.eval()

	for x in _alex:
		
		print("Index : {}".format(total))

		X, q, y = x[0], x[1][0], x[2][0]
		tokens = tks.tokenize(q)
		ids = torch.tensor(tks.convert_tokens_to_ids(tokens)).unsqueeze(0)
		embeds = bert_model(ids)[0]
		flag = 0
		# print (x)
		#print (q)
		for token in tokens:
			if '##' in token:
				flag = 1
				break

		if flag:
			issue += 1

		total += 1
		break
	print (embeds)
	print (embeds.shape)
	# print (total)
		# print ('Hey')
		# print("Question : {}".format(q))
		# print("Answer : {}".format(y))
	# break


