import torch
from torch.utils import data
from torchvision import transforms
from PIL import Image
import json
import os
import numpy as np
import tqdm
from transformers import BertTokenizer, BertModel
import pickle

class CustomDataset(data.Dataset):
	'Characterizes a dataset for PyTorch'
	def __init__(self, data_path, ID_path, json_path, tokens_path, target_image_size, set_="train"):
		'Initialization'
		self.data_path = data_path
		self.cleaned_json = self.read_Json(json_path)

		self.list_IDs = self.read_IDs(ID_path)
		self.ocr_dict = self.read_OCR_tokens(tokens_path)
		# self.list_IDs = [i for i in range(len(self.map_IDs))]
		
		self.target_image_size = target_image_size
		self.set_ = set_
		self.channel_mean = [0.485, 0.456, 0.406]
		self.channel_std = [0.229, 0.224, 0.225]

		self.color_transforms = transforms.Compose([
				transforms.Resize(self.target_image_size),
				transforms.ToTensor(),
				transforms.Normalize(self.channel_mean, self.channel_std)])

		self.answer_dict = self.read_answer_vocab()

	def read_OCR_tokens(self, file):

		ocr_dict = {}
		ans = []
		with open(file,"r") as f:
			for line in f:
				line = line.strip()
				eles = line.split(" ")
				id_ = eles[1].split(".")[0]

				num_tokens = int(eles[-1])
				assert(num_tokens == len(eles)-3)

				if num_tokens == 0:
					ocr_dict[id_] = ["NA"]
				else :
					ocr_dict[id_] = eles[2:-1]

		return ocr_dict

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
	
	def get_ID(self, index):
		return self.list_IDs[index]

	def get_path(self, index):
		ID = self.cleaned_json["image_ids"][self.list_IDs[index]]
		return os.path.join(os.getcwd(),os.path.join(self.data_path, self.set_+"/" + self.set_ + "_images/"+ID+".jpg"))

	def read_answer_vocab(self):

		answer_dict = {}
		counter = 0

		with open(os.path.join(os.getcwd(),"Data/answer_space.txt"),"r") as f:
			lines = f.readlines()
			for line in lines :
				line = line.strip()
				line = line.lower()

				if answer_dict.get(line,-1)==-1:
					answer_dict[line] = counter
					counter += 1

		print("Answer Vocab Size : {}".format(len(answer_dict)))
		return answer_dict


	def __getitem__(self, index):
		'Generates one sample of data'

		# Select sample
		# ID = list(self.cleaned_json["image_classes"].keys())[index]
		# Iterating through test set showed that dictionary key "image_ids does not exist"
		
		ID = self.cleaned_json["image_ids"][self.list_IDs[index]]
		
		if self.ocr_dict.get(ID,None) is not None:
			tokens = " ".join(self.ocr_dict[ID])
		else :
			tokens = "NA"

		# print (ID)
		# Load image
		image_path = os.path.join(self.data_path, self.set_+"/" + self.set_ + "_images/"+ID+".jpg")
		X = Image.open(image_path)
		
		if X.mode != 'RGB' :
			X = X.convert('RGB')
			
		# transform = transforms.ToTensor()
		# raw_image = transform(X)
		transformed_image = self.color_transforms(X)

		# print("PIL raw_image Size : {}".format(np.array(X).shape))
		# print("Tensor raw_image Size : {}".format(raw_image.size()))
		# print("Tensor transformed_image Size : {}".format(transformed_image.size()))

		# Load question
		q = self.cleaned_json["question"][self.list_IDs[index]]

		ans = {}
		if self.set_ == "test":
			return image_path,transformed_image,q,tokens
		else :
			# Load Label
			y = self.cleaned_json["answers"][self.list_IDs[index]][0]
			y_idx = self.answer_dict[y.strip().lower()]
			return image_path,transformed_image,q,tokens,y,y_idx

def create_answer_space(file:str,answers:list) :

	for index,ans in enumerate(answers) :
		
		word = ans + "\n"

		try :
			with open(file,"a") as f:
				f.write(word)
		except :
			with open(file,"w") as f:
				f.write(word)


if __name__ == '__main__' :

	mode = "train"
	data_path = "/home/alex/Desktop/4-2/Text_VQA/Data/"
	ID_path = os.path.join(data_path,"{}/{}_ids.txt".format(mode,mode))
	json_path = os.path.join(data_path,"{}/cleaned.json".format(mode))
	tokens_path = os.path.join(data_path,"{}/{}_tokens_in_images.txt".format(mode,mode))

	alex = CustomDataset(data_path,ID_path,json_path,tokens_path,(448,448),mode)
	total = 0
	issue = 0
			
	alex_loader = data.DataLoader(alex,batch_size=64)

	for x in alex_loader:
		
		print("Index : {}".format(alex.list_IDs[total]))

		if mode == "train" or mode == "dev" :
			image_path,transformed_image,q,tokens,y,y_idx = x[0], x[1], x[2], x[3], x[4],x[5]
		else :
			image_path,transformed_image,q,tokens = x[0], x[1], x[2], x[3]

		#print("Image Path : {}".format(image_path))
		print("Image : {}".format(transformed_image.size()))
		#print("Q : {}".format(q))
		#print("Tokens : {}".format(tokens))
		
		if mode == "train" or mode == "dev" :
			print("Y : {}".format(y[:5]))
			print("Idxs : {}".format(y_idx[:5]))


		total += transformed_image.size()[0]

		# create_answer_space("answer_space.txt",list(y))

	# 	tokens = tks.tokenize(q)
	# 	ids = torch.tensor(tks.convert_tokens_to_ids(tokens)).unsqueeze(0)
	# 	embeds = bert_model(ids)[0]
	# 	flag = 0
	# 	# print (x)
	# 	#print (q)
	# 	for token in tokens:
	# 		if '##' in token:
	# 			flag = 1
	# 			break

	# 	if flag:
	# 		issue += 1

	# 	total += 1
	# 	break
	# print (embeds)
	# print (embeds.shape)
	# print (total)
		# print ('Hey')
		# print("Question : {}".format(q))
		# print("Answer : {}".format(y))
	# break


