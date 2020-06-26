from Multimodal import GridFeaturesAndText,ObjectFeaturesAndText,VectorizedOCRTokensAndText,CombineModes
from image_features import GridFeatures,EndToEndFeatExtractor
from Text_Features import BertTextModel
from utils.customDatasets import CustomDataset
from typing import List
from torch.utils import data
import numpy as np
import torch
import os

class GloveEmbeddings(object):
	def __init__(self,max_tokens:int,embed_dim:int,glove_file:str = 'Data/glove.6B.300d.txt'):
		self.MAX_TOKENS = max_tokens
		self.EMBED_DIM = embed_dim
		with open(os.path.join(os.getcwd(),glove_file),'r') as f:
			self.words = set()
			self.word_to_vec_map = {} 
			n = 0           
			unk_token = 0
			for line in f:
				n += 1 
				line = line.strip().split()
				curr_word = line[0]
				self.words.add(curr_word)
				self.word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)
				unk_token += np.array(line[1:], dtype=np.float64)
			self.unk_token = unk_token / n
		assert self.unk_token.shape == np.array(line[1:]).shape

	def get_embedding(self, word):

		word = str(word).lower()
		try:
			embedding = self.word_to_vec_map[word]
		except KeyError:
			embedding = self.unk_token
		return embedding

	def get_sentence_embedding(self,sent_list:List[str]) :

		all_words = []
		all_embeddings = []
		all_lengths = []

		for sent in sent_list:

			# tokens
			word_tokens = [ "pad" for i in range(self.MAX_TOKENS) ]
			actual_tokens = sent.split(" ")
			for idx,token in enumerate(actual_tokens):
				word_tokens[idx] = token.strip().lower()
			all_words.append(word_tokens)

			# embeddings
			word_embeddings = np.zeros((self.MAX_TOKENS,self.EMBED_DIM))
			for idx,word in enumerate(actual_tokens) :
				word_embeddings[idx] = self.get_embedding(word)
			word_embeddings = torch.tensor(word_embeddings)
			all_embeddings.append(word_embeddings)

			# lengths
			all_lengths.append(len(actual_tokens))
		
		all_lengths = torch.tensor(np.array(all_lengths))
		all_embeddings = torch.stack(all_embeddings).float()

		return all_words,all_embeddings,all_lengths


def train(trainLoader,bert,grid_feature_extractor,object_feature_extractor,
	grid_text_attend,object_text_attend,ocr_text_attend,multi_modal,glove):
	
	idx = 0
	for dataItem in trainLoader :

		idx += 1
		x = dataItem

		#image_path,transformed_image,q,tokens,y,y_idx
		image_paths,transformed_images,qs,ocr_token_sents,y,y_idx = x[0], x[1], x[2], x[3], x[4], x[5]

		# extract grid features
		img_grid_fts = grid_feature_extractor(transformed_images)
		print("Grid Features extracted, Size : {}".format(img_grid_fts.size()))

		# extract object features
		img_obj_fts,num_objects = object_feature_extractor(image_paths)
		print("Object Features extracted, Size : {}".format(img_obj_fts.size()))

		# extract the text features
		all_steps,txt_fts = bert(qs)
		print("Text Features extracted, Size : {}".format(txt_fts.size()))

		# extract the OCR features
		ocr_words,ocr_embeddings,ocr_lengths = glove.get_sentence_embedding(ocr_token_sents)
		print("OCR Features extracted")

		# attention between the grid features and text features
		grid_text = grid_text_attend(txt_fts,img_grid_fts)
		print("Grid and Text Attention Applied, Size : {}".format(grid_text.size()))

		# attention between the object features and text features
		obj_text = object_text_attend(txt_fts,img_obj_fts,num_objects)
		print("Object and Text Attention Applied, Size : {}".format(obj_text.size()))

		# attention between the ocr features and text features
		ocr_text = ocr_text_attend(txt_fts,ocr_embeddings,ocr_lengths)
		print("OCR and Text Attention Applied, Size : {}".format(ocr_text.size()))

		# final combination
		predictions = multi_modal(txt_fts,grid_text,obj_text,ocr_text,ocr_embeddings,ocr_lengths)
		print("Prediction Made")

		if idx > 20:
			return predictions

	return

if __name__ == '__main__' :

	# Dataloaders
	mode = "train"
	data_path = "/home/alex/Desktop/4-2/Text_VQA/Data/"
	train_ID_path = os.path.join(data_path,"{}/{}_ids.txt".format(mode,mode))
	train_json_path = os.path.join(data_path,"{}/cleaned.json".format(mode))
	tokens_path = os.path.join(data_path,"tokens_in_images.txt")
	trainDataset = CustomDataset(data_path,train_ID_path,train_json_path,tokens_path,(448,448),mode)
	trainLoader = data.DataLoader(trainDataset,batch_size=2)

	# dev_ID_path = os.path.join(data_path,"dev/dev_ids.txt")
	# dev_json_path = os.path.join(data_path,"dev/cleaned.json")
	# devDataset = CustomDataset(data_path,dev_ID_path,
	# 				dev_json_path,(448,448),"dev")
	# devLoader = data.DataLoader(devDataset)

	# Feature Extractors
	print("Loading Bert")
	bert = BertTextModel("/home/alex/Desktop/4-2/Text_VQA/utils/output/").eval()
	print("\tBert Initialized\n")
	
	print("Loading ResNet-101")
	grid_feature_extractor = GridFeatures("resnet101").eval()
	print("\tResNet-101 Initialized\n")

	print("Loading Faster R-CNN")
	object_feature_extractor = EndToEndFeatExtractor().eval()
	print("\tFaster R-CNN Initialized\n")

	# Attention Models
	VOCAB_SIZE = 17695
	MAX_TOKENS = 50
	QUEST_DIM = 768
	IMG_DIM = (2048,14,14)
	MAX_OBJECTS = 50
	EMBED_DIM = 300
	OCR_DIM = 300

	print("Loading GridFeaturesAndText")
	grid_text_attend = GridFeaturesAndText(QUEST_DIM,IMG_DIM)
	print("\tGridFeaturesAndText Initialized\n")

	print("Loading ObjectFeaturesAndText")
	object_text_attend = ObjectFeaturesAndText(QUEST_DIM,IMG_DIM,MAX_OBJECTS)
	print("\tObjectFeaturesAndText Initialized\n")

	print("Loading VectorizedOCRTokensAndText")
	ocr_text_attend = VectorizedOCRTokensAndText(QUEST_DIM,EMBED_DIM)
	print("\tVectorizedOCRTokensAndText Initialized\n")

	# Final Combination Module
	print("Loading CombineModes")
	multi_modal = CombineModes(QUEST_DIM,OCR_DIM,IMG_DIM,VOCAB_SIZE,EMBED_DIM,MAX_TOKENS)
	print("\tCombineModes Initialized\n")

	# Glove
	print("Loading GloveEmbeddings")
	glove = GloveEmbeddings(MAX_TOKENS,EMBED_DIM)
	print("\tGloveEmbeddings Initialized\n")

	predictions = train(trainLoader,bert,grid_feature_extractor,object_feature_extractor,
					grid_text_attend,object_text_attend,ocr_text_attend,multi_modal,glove)
	print("Predictions : {}".format(predictions.size()))

	# question = ["I know you","We know that"]
	# image = torch.randn((2,3,448,448))

	# all_steps,text_ft = bert.forward(question)
	# print(text_ft.size())

	# img_ft = resnet(image)
	# print(img_ft.size())

	# embed_dim = text_ft.size()[1]

	# GFT = GridFeaturesAndText(embed_dim,img_ft.size()[1:])

	# ans = GFT.forward(text_ft,img_ft)
	# print(ans.size())




