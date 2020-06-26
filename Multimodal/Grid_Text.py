import torch
import torch.nn as nn
import torch.nn.functional as F

class GridFeaturesAndText(nn.Module):
	""" Combine Visual and Text Features """

	def __init__(self,quest_dim:int,img_ft_dims:list) -> None:
		"""
		Arguments :
			quest_dim : The dimensionality for the question embedding
						ex. 768 for BERT
			img_ft_dims : The size of the image features from the feature extractor
		"""
		super(GridFeaturesAndText,self).__init__()

		self.QUEST_DIM = quest_dim
		self.img_ft_c = img_ft_dims[0]
		self.img_ft_x = img_ft_dims[1]
		self.img_ft_y = img_ft_dims[2]

		self.net1 = nn.Linear(self.QUEST_DIM,self.img_ft_c)

	def forward(self,text_fts:torch.tensor,img_fts:torch.tensor) -> torch.tensor:
		
		# text_fts : (batch_size,QUEST_DIM)
		batch_size = text_fts.size()[0]

		# out1 : (batch_size,img_ft_c)
		out1 = self.net1(text_fts)

		# out1 : (batch_size,img_ft_c,1,1)
		out1 = out1.unsqueeze(2).unsqueeze(3)
		#print(out1.size())

		# similarities : (batch_size,img_ft_c,img_ft_x,img_ft_y)
		similarities = out1*img_fts
		#print(similarities.size())

		# similarities : (batch_size,img_ft_x,img_ft_y)
		similarities = torch.sum(similarities,dim=1)
		#print(similarities.size())

		# similarities : (batch_size,img_ft_x*img_ft_y)
		similarities = similarities.view(batch_size,-1)
		#print(similarities.size())

		# wights : (batch_size,img_ft_x*img_ft_y)
		weights = F.softmax(similarities,dim=1)
		#print(weights.size())

		# img_fts : (batch_size,img_ft_c,img_ft_x,img_ft_y)
		# final_ans : (batch_size,img_ft_c,1)
		final_ans = torch.bmm(img_fts.view(batch_size,self.img_ft_c,self.img_ft_x*self.img_ft_y),weights.unsqueeze(2))
		#print(final_ans.size())

		# final_ans : (batch_size,img_ft_c)
		final_ans = final_ans.squeeze(2)
		#print(final_ans.size())

		return final_ans


class AskAttendAnswer(nn.Module):
	""" Ask Attend Answer Attention Mechanism """

	def __init__(self,embed_dim:int,img_ft_dims:tuple,batch_size:int,max_seq_len:int,num_outs:int):
		
		super(AskAttendAnswer).__init__()

		self.embed_dim = embed_dim
		self.img_ft_c = img_ft_dims[0]
		self.img_ft_x = img_ft_dims[1]
		self.img_ft_y = img_ft_dims[2]
	
		self.max_seq_len = max_seq_len
		self.num_outs = num_outs

		# Attention Weights
		self.W_A = nn.init.xavier_uniform_(torch.empty(self.img_ft_c,self.embed_dim) , gain=nn.init.calculate_gain('relu'))
		self.b_A = nn.init.xavier_uniform_(torch.empty(self.img_ft_x*self.img_ft_y,self.embed_dim), gain=nn.init.calculate_gain('relu'))

		# Evidence Weights
		self.W_E = nn.init.xavier_uniform_(torch.empty(self.img_ft_c,self.embed_dim) , gain=nn.init.calculate_gain('relu'))
		self.b_E = nn.init.xavier_uniform_(torch.empty(self.img_ft_x*self.img_ft_y,self.embed_dim), gain=nn.init.calculate_gain('relu'))

		# Word Weights
		self.W_Q = nn.init.normal_(torch.empty(self.max_seq_len))
		self.b_Q = nn.init.normal_(torch.empty(self.embed_dim))

		# Prediction Weights
		self.W_P = nn.init.xavier_uniform_(torch.empty(self.embed_dim,self.num_outs) , gain=nn.init.calculate_gain('relu'))
		self.b_P = nn.init.normal_(torch.empty(self.num_outs))

	def combine(self,text_fts:torch.tensor,img_fts:torch.tensor) -> torch.tensor:

		assert((self.batch_size,self.max_seq_len,self.embed_dim) == text_fts.size())

		# self.batch_size,self.img_ft_c,self.img_ft_x,self.img_ft_y
		# TO
		# self.batch_size,self.img_ft_y,self.img_ft_x,self.img_ft_c
		# TO
		# self.batch_size,self.img_ft_x,self.img_ft_y,self.img_ft_c

		img_fts = torch.transpose(img_fts,1,3)
		img_fts = torch.transpose(img_fts,1,2)

		assert((self.batch_size,self.img_ft_x,self.img_ft_y,self.img_ft_c) == img_fts.size())

		img_fts = img_fts.view((self.batch_size,self.img_ft_x*self.img_ft_y,self.img_ft_c))

		# Correlation Matrix
		self.correlation_mat = torch.matmul( torch.matmul(img_fts,self.W_A)+self.b_A, torch.transpose(text_fts,1,2) )
		print("Correlation Mat : {}".format(self.correlation_mat.size()))
		assert((self.batch_size,self.img_ft_x*self.img_ft_y,self.max_seq_len) == self.correlation_mat.size())


		# Attention Matrix
		self.W_att = torch.max(F.softmax(self.correlation_mat,dim=2),dim=2)[0]
		print("Attention Mat : {}".format(self.W_att.size()))
		assert((self.batch_size,self.img_ft_x*self.img_ft_y) == self.W_att.size())


		# Evidence Vector
		self.S_att = torch.bmm( torch.unsqueeze(self.W_att,dim=1) , torch.matmul(img_fts,self.W_E)+self.b_E ).squeeze(dim=1)
		print("Evidence Vector : {}".format(self.S_att.size()))
		assert((self.batch_size,self.embed_dim) == self.S_att.size())
		#print("Left : {}".format(self.W_att.size()))
		#print("Right : {}".format((torch.matmul(img_fts,self.W_E)+self.b_E).size()))


		# Question Embedding
		self.Q = torch.matmul(self.W_Q,text_fts).squeeze(dim=1) + self.b_Q
		print("Question Embedding : {}".format(self.Q.size()))
		#print("Left : {}".format(self.W_Q.size()))
		#print("Right : {}".format((text_fts).size()))


		# Final Predictions
		predictions = torch.matmul( F.relu(self.S_att+self.Q), self.W_P )+self.b_P
		print("Predictions Size : {}".format(predictions.size()))
		#print("Left : {}".format(self.W_P.size()))
		#print("Right : {}".format(self.S_att.size()))

		return predictions