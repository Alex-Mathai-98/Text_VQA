from Multimodal import GridFeaturesAndText,ObjectFeaturesAndText,VectorizedOCRTokensAndText,CombineModes
from image_features import GridFeatures,EndToEndFeatExtractor
from Text_Features import BertTextModel
from utils.customDatasets import CustomDataset
from typing import List
from torch.optim import Adam
from torch.utils import data
import numpy as np
import torch
import os
import gc

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

			#sent = list(sent)

			print("Sent : {}".format(sent))

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

class Trainer() :

	def __init__(self,config):

		self.trainLoader = config["trainLoader"]
		self.text_feature_extractor = config["text_feature_extractor"]
		self.grid_feature_extractor = config["grid_feature_extractor"]
		self.object_feature_extractor = config["object_feature_extractor"]
		self.VOCAB_SIZE = config["vocab_size"]+1
		self.MAX_TOKENS = config["max_tokens"]
		self.QUEST_DIM = config["quest_dim"]
		self.IMG_DIM = config["img_dim"]
		self.MAX_OBJECTS = config["max_objects"]
		self.EMBED_DIM = config["embed_dim"]
		self.OCR_DIM = config["ocr_dim"]

		print("Loading GridFeaturesAndText")
		self.grid_text_attend = GridFeaturesAndText(self.QUEST_DIM,self.IMG_DIM)
		
		print("\tGridFeaturesAndText Initialized\n")

		print("Loading ObjectFeaturesAndText")
		self.object_text_attend = ObjectFeaturesAndText(self.QUEST_DIM,self.IMG_DIM,self.MAX_OBJECTS)
		print("\tObjectFeaturesAndText Initialized\n")

		print("Loading VectorizedOCRTokensAndText")
		self.ocr_text_attend = VectorizedOCRTokensAndText(self.QUEST_DIM,self.EMBED_DIM)
		print("\tVectorizedOCRTokensAndText Initialized\n")

		# Final Combination Module
		print("Loading CombineModes")
		self.multi_modal = CombineModes(self.QUEST_DIM,self.OCR_DIM,self.IMG_DIM,self.VOCAB_SIZE,self.EMBED_DIM,self.MAX_TOKENS)
		print("\tCombineModes Initialized\n")

		# Glove
		print("Loading GloveEmbeddings")
		self.glove = GloveEmbeddings(self.MAX_TOKENS,self.EMBED_DIM)
		print("\tGloveEmbeddings Initialized\n")

		self.optim = Adam([ {'params' : self.grid_text_attend.parameters()},
							{'params' : self.object_text_attend.parameters()},
							{'params' : self.ocr_text_attend.parameters()},
							{'params' : self.multi_modal.parameters()} ],lr=5e-4)

	def get_copy_log_mask(self,y,ocr_words):

		if type(ocr_words) != np.ndarray:
			ocr_words = np.array(ocr_words)
		if type(y) != np.ndarray:
			y = np.array(y)
			y = np.expand_dims(y,axis=1)

		#print(ocr_words)
		#print(y)
		# Create mask by checking where the answer matches the label
		copy_mask = (ocr_words==y) 
		copy_mask = torch.tensor(copy_mask).float()
		assert(copy_mask.size()==(len(y),self.MAX_TOKENS))

		# Take a log of the mask
		#copy_log_mask = torch.log(copy_mask + 1e-45)
		#assert(copy_log_mask.size()[1]==self.MAX_TOKENS)
		return copy_mask

	def get_target_log_mask(self,y_idx,in_ocr):

		# fill everything with almost 0
		vocab_mask = torch.zeros((len(y_idx),self.VOCAB_SIZE))
		
		print("y_idx : {}".format(y_idx))
		# make ground truth mask as 1 for the correct labels
		vocab_mask[torch.arange(0,len(y_idx)),y_idx] = 1

		# if the answer is in the OCR tokens - make the ground_truth mask as 0
		# to avoid contribution from answer space
		if torch.sum(in_ocr)==0:
			print("No answers in the OCR")
		else :
			print("Old Max : {}, Old Min : {}".format(torch.max(vocab_mask[in_ocr]), torch.min(vocab_mask[in_ocr])))
			vocab_mask[in_ocr,:] = torch.zeros((self.VOCAB_SIZE))
			print("New Max : {}, New Min : {}".format(torch.max(vocab_mask[in_ocr]), torch.min(vocab_mask[in_ocr])))

		# taking log
		# vocab_log_mask = torch.log(vocab_mask + 1e-45)
		return vocab_mask

	def loss_function(self,predictions,y,y_idx,in_ocr,ocr_words):

		in_ocr = in_ocr.byte()

		#print("predictions[]")
		#predictions = torch.log(predictions + 1e-45)

		# Note 1 : log(prob) + log(1e-45) = log(prob*1e-45)
		# Note 2 : exp( log(prob*1e-45) ) ~ 0

		# Note 3 : log(prob) + log(1) = log(prob)
		# Note 4 : exp( log(prob ) = prob

		# This is what we are doing, the log(1) and log(1e-45) are created with the mask arrs
		# Using this mask, we get the prediction indices that are important
		# We add up the probabilities and then take the loss function

		copy_log_mask = self.get_copy_log_mask(y,ocr_words)
		copy_log_probs = predictions[:,self.VOCAB_SIZE:] * copy_log_mask

		vocab_log_mask = self.get_target_log_mask(y_idx,in_ocr)
		vocab_log_probs = predictions[:,:self.VOCAB_SIZE] * vocab_log_mask

		final_probs = torch.cat((vocab_log_probs,copy_log_probs),dim=1)
		try :
			print("Max 1 : {}".format(torch.max(final_probs,dim=1)[0].detach().cpu().numpy()))
			print("Indices 1 : {}".format(torch.max(final_probs,dim=1)[1].detach().cpu().numpy()))
		except :
			print(final_probs)
			assert(0)

		max_idxs = torch.max(final_probs,dim=1)[1]
		for idx,truth,guess in zip(range(len(y_idx)),y_idx,max_idxs) :
			if truth != guess and in_ocr[idx]==0 :
				print("Truth : {}, Guess : {}".format(final_probs[idx,truth],final_probs[idx,guess]))

		# final_probs = torch.exp(final_probs)
		# print("Max 2 : {}".format(torch.max(final_probs,dim=1)[0]))
		# print("Indices 2 : {}".format(torch.max(final_probs,dim=1)[1]))

		final_probs = torch.sum(final_probs,dim=1)

		print("Final Probs Vector : {}".format(final_probs))
		loss = -torch.log(final_probs+1e-45)
		print("Loss Vector : {}".format(loss))

		loss = torch.sum(loss)/len(loss)

		return loss

	def train(self):
		
		idx = 0
		for dataItem in self.trainLoader :

			idx += 1
			x = dataItem

			self.optim.zero_grad()

			self.grid_text_attend.train()
			self.object_text_attend.train()
			self.ocr_text_attend.train()
			self.multi_modal.train()

			#image_path,transformed_image,q,tokens,y,y_idx
			image_paths,transformed_images,qs,ocr_token_sents,y,y_idx,in_ocr = x[0], x[1], x[2], x[3], x[4], x[5], x[6]

			# extract grid features
			img_grid_fts = self.grid_feature_extractor(transformed_images)
			print("Grid Features extracted, Size : {}".format(img_grid_fts.size()))

			# extract object features
			img_obj_fts,num_objects = self.object_feature_extractor(image_paths)
			print("Object Features extracted, Size : {}".format(img_obj_fts.size()))

			# extract the text features
			all_steps,txt_fts = self.text_feature_extractor(qs)
			print("Text Features extracted, Size : {}".format(txt_fts.size()))

			# extract the OCR features
			# print("Raw : {}".format(ocr_token_sents))
			# print("Type : {}".format(type(ocr_token_sents)))
			# print("A0 : {}".format(ocr_token_sents[0]))
			# print("List A0 : {}".format(list(ocr_token_sents[0])))
			# print("A1 : {}".format(ocr_token_sents[1]))
			# print("A2 : {}".format(ocr_token_sents[2]))
			# print("A3 : {}".format(ocr_token_sents[3]))
			# print(ocr_token_sents)
			# ocr_token_sents = list(ocr_token_sents[0])
			# temp = []
			# for i in range(len(ocr_token_sents)):
			# 	temp.append(ocr_token_sents[i])
			# ocr_token_sents = temp
			# print("Num1 : {}".format(ocr_token_sents))
			print(ocr_token_sents)
			print(in_ocr)
			ocr_words,ocr_embeddings,ocr_lengths = self.glove.get_sentence_embedding(ocr_token_sents)
			print("OCR Features extracted, Size : {}".format(ocr_embeddings.size()))

			# attention between the grid features and text features
			grid_text = self.grid_text_attend(txt_fts,img_grid_fts)
			print("Grid and Text Attention Applied, Size : {}".format(grid_text.size()))

			# attention between the object features and text features
			obj_text = self.object_text_attend(txt_fts,img_obj_fts,num_objects)
			print("Object and Text Attention Applied, Size : {}".format(obj_text.size()))

			# attention between the ocr features and text features
			ocr_text = self.ocr_text_attend(txt_fts,ocr_embeddings,ocr_lengths)
			print("OCR and Text Attention Applied, Size : {}".format(ocr_text.size()))

			# final combination
			predictions = self.multi_modal(txt_fts,grid_text,obj_text,ocr_text,ocr_embeddings,ocr_lengths)
			print("Prediction Made")

			loss = self.loss_function(predictions,y,y_idx,in_ocr,ocr_words)
			print("Calculated Loss : {}\n".format(loss.item()))

			loss.backward()
			self.optim.step()

			if idx > 20:
				return predictions

			torch.cuda.empty_cache()
			gc.collect()
			#assert(False)

		return

if __name__ == '__main__' :

	# Dataloaders
	mode = "train"
	data_path = "/home/alex/Desktop/4-2/Text_VQA/Data/"
	train_ID_path = os.path.join(data_path,"{}/{}_ids.txt".format(mode,mode))
	train_json_path = os.path.join(data_path,"{}/cleaned.json".format(mode))
	tokens_path = os.path.join(data_path,"tokens_in_images.txt")
	trainDataset = CustomDataset(data_path,train_ID_path,train_json_path,tokens_path,(448,448),mode)
	trainLoader = data.DataLoader(trainDataset,batch_size=4)

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

	config = {}
	config["trainLoader"] = trainLoader
	config["text_feature_extractor"] = bert
	config["grid_feature_extractor"] = grid_feature_extractor
	config["object_feature_extractor"] = object_feature_extractor
	config["vocab_size"] = VOCAB_SIZE
	config["max_tokens"] = MAX_TOKENS
	config["quest_dim"] = QUEST_DIM
	config["img_dim"] = IMG_DIM
	config["max_objects"] = MAX_OBJECTS
	config["embed_dim"] = EMBED_DIM
	config["ocr_dim"] = OCR_DIM
	
	alex = Trainer(config)
	alex.train()

	