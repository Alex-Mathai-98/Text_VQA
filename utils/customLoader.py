import torch
from torch.utils import data
from torchvision import transforms
from PIL import Image
import json
import os

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

		self.transforms = transforms.Compose([
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

	def __getitem__(self, index):
		'Generates one sample of data'

		# Select sample
		ID = self.list_IDs[index]

		# Load data and get label
		X = Image.open(os.path.join(data_path,self.set_+"/" + self.set_ + "_images/"+ID+".jpg"))
		X = self.transforms(X)

		if self.set_ == "test" :
			return X

		y = self.cleaned_json["answers"][ID][0]

		return X, y


if __name__ == '__main__' :

	data_path = "/home/alex/Desktop/4-2/Text_VQA/Data/"
	ID_path = os.path.join(data_path,"test/test_ids.txt")
	json_path = os.path.join(data_path,"test/cleaned.json")
	alex = CustomDataset(data_path,ID_path,json_path,(448,448),set_="test")

	X = alex.__getitem__(2)
	print(X.size())