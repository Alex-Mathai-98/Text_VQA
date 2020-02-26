from Multimodal import AskAttendAnswer
from Image_Features import GridFeatures
from Text_Features import TextModel
from utils import CustomDataset
import os
import torch


if __name__ == '__main__' :

	data_path = "/home/alex/Desktop/4-2/Text_VQA/Data/"
	ID_path = os.path.join(data_path,"train/train_ids.txt")
	json_path = os.path.join(data_path,"train/cleaned.json")
	dataset = CustomDataset(data_path,ID_path,json_path,(448,448),set_="train")

	# Image Extractor
	resnet = GridFeatures("resnet101")
	# Text Extractor
	bert = TextModel()

	#print("Check resnet : {}".format(resnet(torch.randn((1,3,448,448))).size()))
	print("Check bert : {}".format(bert(torch.tensor([1,2,4,0,0])).size()))

	# embed_dim,img_ft_dims,batch_size,max_seq_len,num_outs
	mixer = AskAttendAnswer(300,(2048,14,14),10,64,15000)

	text_fts = torch.nn.init.normal_(torch.empty(10,64,300))
	img_fts = torch.nn.init.normal_(torch.empty(10,3,448,448))

	mixer.combine(text_fts,img_fts)

