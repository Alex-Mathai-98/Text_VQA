import torch
import torch.nn as nn
import torch.nn.functional as F

class CombineModes(nn.Module):

	def __init__(self,quest_dim:int,ocr_dim:int,img_ft_dims:int,vocab_dim:int) -> None:
		"""
		Arguments :
			quest_dim : The dimensionality for the question embedding
						ex. 768 for BERT
			ocr_dim : The dimensionality for the OCR tokens

			img_ft_dims : The dimensionality of the object/grid features after applying the attention
						ex. 2048 for R-CNN 
		"""

		super(FinalCombination,self).__init__()
		self.QUEST_DIM = quest_dim
		self.OCR_DIM = ocr_dim
		self.IMG_DIM = img_ft_dims
		self.VOCAB_DIM = vocab_size

		# To project question embeddding to IMG_DIM
		self.net1 = nn.Linear(self.QUEST_DIM,self.IMG_DIM)

		# To project question embeddding to IMG_DIM
		self.net2 = nn.Linear(self.QUEST_DIM,self.IMG_DIM)

		# To project question embeddding & OCR embedding to IMG_DIM
		self.net3 = nn.Linear(self.QUEST_DIM,self.IMG_DIM)
		self.net4 = nn.Linear(self.OCR_DIM,self.IMG_DIM)

		# To project Combined embedding to Vocab Size
		self.net5 = nn.Linear(self.IMG_DIM,self.VOCAB_DIM)



	def forward(self,text_fts:torch.tensor,img_fts:torch.tensor,obj_fts:torch.tensor,ocr_fts:torch.tensor):
		""" Takes all the contextual embeddings from the question, image and OCR to answer the question. 

		Arguments :
			text_fts : (m,QUEST_DIM). Text features
			img_fts : (m,IMG_DIM). Image Grid features
			obj_fts : (m,IMG_DIM). Image Object features
			ocr_fts : (m,OCR_DIM). OCR Token features
		"""

		out1 = self.net1(text_fts)
		question_grid = F.sigmoid(out1,dim=1)*img_fts

		out2 = self.net1(text_fts)
		question_obj = F.sigmoid(out2,dim=1)*img_fts

		out3 = self.net3(text_fts)
		out4 = self.net4(ocr_fts)
		question_ocr = F.sigmoid(out3,dim=1)*out4

		final_combined = question_grid + question_obj + question_ocr

		prediction = self.net5(final_combined)

		return prediction