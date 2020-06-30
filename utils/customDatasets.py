import torch
from torch.utils import data
from torchvision import transforms
from PIL import Image
import json
import os
from typing import List
import numpy as np
import tqdm
from transformers import BertTokenizer, BertModel
import pickle

class CustomDataset(data.Dataset):
	'Characterizes a dataset for PyTorch'
	def __init__(self, data_path, ID_path, json_path, tokens_path, target_image_size, set_="train",max_tokens=50):
		'Initialization'

		self.MAX_TOKENS = max_tokens
		self.data_path = data_path
		self.cleaned_json = self.read_Json(json_path)

		self.list_IDs = self.read_IDs(ID_path)
		self.ocr_dict = self.read_OCR_tokens(tokens_path)
		
		self.target_image_size = target_image_size
		self.set_ = set_
		self.channel_mean = [0.485, 0.456, 0.406]
		self.channel_std = [0.229, 0.224, 0.225]

		self.color_transforms = transforms.Compose([
				transforms.Resize(self.target_image_size),
				transforms.ToTensor(),
				transforms.Normalize(self.channel_mean, self.channel_std)])

		self.answer_dict = self.read_answer_vocab()
		self.OCR_Answer_Overlap()

	def OCR_Answer_Overlap(self):

		image_ids = list(self.id_to_answer.keys())

		exists_in_OCR = 0
		total_ids = 0

		answers = []

		for image in image_ids :
			
			ans = self.id_to_answer[image].strip().lower()
			
			tokens = self.ocr_dict[image]
			tokens = [tok.strip().lower() for tok in tokens]

			if ans in tokens:
				exists_in_OCR += 1
				answers.append(ans)

			total_ids += 1

		# print("Total Previous Size : {}".format(len(self.answer_dict)))
		# for ans in answers :
		# 	# if count is 1, then remove it
		# 	if self.answer_dict[ans][1] == 1:
		# 		del self.answer_dict[ans]
		# 	else :
		# 		self.answer_dict[ans][1] -= 1
		# print("New Size : {}".format(len(self.answer_dict)))

		print("Percentage Sucessful : {}".format(float(exists_in_OCR)*100/float(total_ids)))

		return

	def read_OCR_tokens(self, file):

		ocr_dict = {}
		ans = []

		total_input = 0
		no_ocr = 0

		with open(file,"r") as f:
			for line in f:
				line = line.strip()
				eles = line.split(" ")
				id_ = eles[1].split(".")[0]

				num_tokens = int(eles[-1])
				assert(num_tokens == len(eles)-3)

				if num_tokens == 0:
					ocr_dict[id_] = [""]
					no_ocr += 1
				else :
					ocr_dict[id_] = eles[2:-1]
				total_input += 1

		print("Total : {}".format(total_input))
		print("No OCR : {}".format(no_ocr))
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

		id_to_answer = {}
		answer_dict = {}
		counter = 0

		with open(os.path.join(os.getcwd(),"Data/new_answer_space.txt"),"r") as f:
			lines = f.readlines()
			for line in lines :
				line = line.strip()
				line = line.lower()

				image_id = line.split(" ")[0]
				answer = " ".join(line.split(" ")[1:])

				id_to_answer[image_id] = answer

				if answer_dict.get(answer,-1)==-1:
					answer_dict[answer] = [counter,1]
					counter += 1
				else :
					answer_dict[answer][1] += 1

		answer_dict["unk"] = [counter,1]
		counter += 1

		print("Answer Vocab Size : {}".format(len(answer_dict)))
		self.id_to_answer = id_to_answer
		return answer_dict


	def get_answer_idx(self,tokens:List[str],y:str):

		try :
			assert(type(tokens)==List[str])
		except :
			if type(tokens) == str :
				tokens = tokens.split(" ")
			else :
				assert(False)

		print("get_answer_idx ==> tokens 1 : {}".format(tokens))
		print("get_answer_idx ==> y : {}".format(y))

		y = y.strip().lower()
		tokens = [tok.strip().lower() for tok in tokens]

		if y in tokens:
			# answer in OCR tokens - so target idx will be "unk"
			y_idx = self.answer_dict["unk"][0]
			# make sure that the answer is in the first "MAX_TOKENS" of the list
			if len(tokens) > self.MAX_TOKENS :
				tokens[randint(0,self.MAX_TOKENS)] = y
		else :
			# answer not in OCR - so target idx will be part of answer space
			y_idx = self.answer_dict[y][0]

		# cut the list short
		tokens = tokens[:self.MAX_TOKENS]
		tokens = " ".join(tokens)
		print("get_answer_idx ==> tokens 2 : {}\n".format(tokens))		

		if y_idx == self.answer_dict["unk"][0] :
			return tokens,y_idx,1
		else :
			return tokens,y_idx,0


	def __getitem__(self, index):
		'Generates one sample of data'

		# Select sample
		# ID = list(self.cleaned_json["image_classes"].keys())[index]
		# Iterating through test set showed that dictionary key "image_ids does not exist"
		
		ID = self.cleaned_json["image_ids"][self.list_IDs[index]]
		
		if self.ocr_dict.get(ID,None) is not None:
			tokens = " ".join(self.ocr_dict[ID])
		else :
			tokens = ""

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

			# preprocessing function
			tokens,y_idx,in_ocr = self.get_answer_idx(tokens,y)

			return image_path,transformed_image,q,tokens,y,y_idx,in_ocr

def create_answer_space(file:str,image_paths:list,answers:list) :

	for path,ans in zip(image_paths,answers) :
		
		image_id = path.split("/")[-1].split(".")[0]
		print("Image Id : {}, Answer : {}".format(image_id,ans))

		word = image_id + " " + ans + "\n"
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
	tokens_path = os.path.join(data_path,"tokens_in_images.txt")

	alex = CustomDataset(data_path,ID_path,json_path,tokens_path,(448,448),mode)
	total = 0
	issue = 0
	#assert(0)
			
	alex_loader = data.DataLoader(alex,batch_size=64)

	for idx,x in enumerate(alex_loader):
		
		print("Index : {}".format(alex.list_IDs[total]))

		if mode == "train" or mode == "dev" :
			image_paths,transformed_images,qs,tokens,ys,y_idxs,in_ocr = x[0], x[1], x[2], x[3], x[4],x[5],x[6]
		else :
			image_paths,transformed_images,qs,tokens = x[0], x[1], x[2], x[3]

		#print("Image Path : {}".format(image_paths))
		print("Image : {}".format(transformed_images.size()))
		print("Q : {}".format(qs))
		#print("Tokens : {}".format(tokens))
		
		if mode == "train" or mode == "dev" :
			print("In OCR : {}".format(torch.sum(in_ocr)))
			print("Questions : {}".format(np.array(list(qs))[in_ocr][:5]))
			print("Y : {}".format(np.array(list(ys))[in_ocr][:5]))
			print("Idxs : {}".format(y_idxs[:5]))


		total += transformed_images.size()[0]

		if idx > 10:
			break

		#create_answer_space("new_answer_space.txt",image_paths,list(ys))

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


