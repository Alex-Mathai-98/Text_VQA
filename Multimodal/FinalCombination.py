import torch
import torch.nn as nn
import torch.nn.functional as F

class CopyNet(nn.Module) :

	def __init__(self,combined_dim:int,embed_dim:int,max_tokens:int) -> None :
		
		super(CopyNet,self).__init__()
		self.COMBINED_DIM = combined_dim
		self.EMBED_DIM = embed_dim
		self.MAX_TOKENS = max_tokens

		# To project the Combined embedding to OCR token embedding size
		self.net1 = nn.Linear(self.COMBINED_DIM,self.EMBED_DIM)

	def loop_forward(self,combined_fts:torch.tensor,ocr_tokens:torch.tensor) -> torch.tensor:
		""" Performs the Copy Net Mechanism using a loop

		Arguments:

			combined_fts : (COMBINED_DIM)
			ocr_tokens : (num_tokens,EMBED_DIM)
		"""

		num_tokens = ocr_tokens.size()[0]

		# out1 : (EMBED_DIM,1)
		out1 = self.net1(combined_fts).unsqueeze(1)
		assert(out1.size()==(self.EMBED_DIM,1))

		# output : (num_tokens)
		output = torch.matmul(ocr_tokens,out1).squeeze(1)
		assert(output.size()[0]==(num_tokens))

		# add the dummy tokens
		dummy_input = torch.ones((self.MAX_TOKENS-num_tokens))
		dummy_input[:] = -1*float("Inf")
		output = torch.cat( (output, dummy_input),dim=0)

		return output

	def forward(self,combined_fts:torch.tensor,ocr_tokens:torch.tensor,num_tokens:torch.tensor) -> torch.tensor :
		""" Performs the Vectorized Copy Net Mechanism.

		Arguments:

			combined_fts : (m,COMBINED_DIM)
			ocr_tokens : (m,MAX_TOKENS,EMBED_DIM)
			num_tokens : (m)

		"""

		BATCH_SIZE = combined_fts.size()[0]

		attention_mask = torch.arange(self.MAX_TOKENS)[None, :] < num_tokens[:, None]

		# out1 : (m,EMBED_DIM,1)
		out1 = self.net1(combined_fts).unsqueeze(2)
		assert(out1.size()==(BATCH_SIZE,self.EMBED_DIM,1))

		# output : (m,MAX_TOKENS)
		output = torch.bmm(ocr_tokens,out1).squeeze(2)
		assert(output.size()==(BATCH_SIZE,self.MAX_TOKENS))

		# mask the dummy tokens
		output[attention_mask==0] = -1*float("Inf")

		return output

class CombineModes(nn.Module):

	def __init__(self,quest_dim:int,ocr_dim:int,img_ft_dims:list,vocab_dim:int,embed_dim:int,max_tokens:int) -> None:
		"""
		Arguments :
			quest_dim : The dimensionality for the question embedding
						ex. 768 for BERT
			ocr_dim : The dimensionality for the OCR tokens

			img_ft_dims : The dimensionality of the object/grid features after applying the attention
						ex. 2048 for R-CNN

			vocab_dim : The vocabulary size

			embed_dim : The dimensionality of the OCR token embeddings
						ex. 300 (Glove)

			max_tokens : The maximum number of OCR tokens allowed in an image
		"""

		super(CombineModes,self).__init__()
		self.QUEST_DIM = quest_dim
		self.OCR_DIM = ocr_dim
		self.IMG_DIM = img_ft_dims[0]
		self.VOCAB_DIM = vocab_dim
		self.EMBED_DIM = embed_dim
		self.MAX_TOKENS = max_tokens

		# To project question embeddding to IMG_DIM ==> Question and Grid Features
		self.net1 = nn.Linear(self.QUEST_DIM,self.IMG_DIM)

		# To project question embeddding to IMG_DIM ==> Question and Object Features
		self.net2 = nn.Linear(self.QUEST_DIM,self.IMG_DIM)

		# To project question embeddding & OCR embedding to IMG_DIM ==> Question and OCR
		self.net3 = nn.Linear(self.QUEST_DIM,self.IMG_DIM)
		self.net4 = nn.Linear(self.OCR_DIM,self.IMG_DIM)

		# To project Combined embedding to Vocab Size ==> Prediction to Answer Space
		self.net5 = nn.Linear(self.IMG_DIM,self.VOCAB_DIM)

		# To project the Combined embedding to OCR token embedding size
		self.copy_net = CopyNet(self.IMG_DIM,self.EMBED_DIM,self.MAX_TOKENS)


	def loop_forward(self,text_fts:torch.tensor,img_fts:torch.tensor,obj_fts:torch.tensor,ocr_fts:torch.tensor,ocr_tokens:torch.tensor) -> torch.tensor:
		""" Takes all the contextual embeddings from the question, image and OCR to answer the question.
		
		Arguments :
			text_fts : (QUEST_DIM). Text features
			img_fts : (IMG_DIM). Image Grid features
			obj_fts : (IMG_DIM). Image Object features
			ocr_fts : (OCR_DIM). OCR Token features
			ocr_tokens : (MAX_TOKENS,EMBED_DIM)
		"""

		# Combine Question and Grid Features
		out1 = self.net1(text_fts) # (IMG_DIM)
		assert(out1.size()[0]== (self.IMG_DIM)), "Print {}, IMG_DIM : {}".format(out1.size(),self.IMG_DIM)
		question_grid = F.sigmoid(out1)*img_fts # (IMG_DIM)
		assert(question_grid.size()[0]==(self.IMG_DIM))

		# Combine Question and Object Features
		out2 = self.net1(text_fts) # (IMG_DIM)
		assert(out2.size()[0]==(self.IMG_DIM))
		question_obj = F.sigmoid(out2)*obj_fts # (IMG_DIM)
		assert(question_obj.size()[0]==(self.IMG_DIM))

		# Combine Question and OCR Features
		out3 = self.net3(text_fts) # (IMG_DIM)
		assert(out3.size()[0]==(self.IMG_DIM))
		out4 = self.net4(ocr_fts) # (IMG_DIM)
		assert(out4.size()[0]==(self.IMG_DIM))
		question_ocr = F.sigmoid(out3)*out4 # (IMG_DIM)
		assert(question_ocr.size()[0]==(self.IMG_DIM))

		# Addition of all 3 vectors
		final_combined = question_grid + question_obj + question_ocr # (IMG_DIM)

		# Linear projection to answer space
		answer_space = self.net5(final_combined) # (VOCAB_DIM)
		assert(answer_space.size()[0]==(self.VOCAB_DIM))

		# Output of copy net
		copy_space = self.copy_net.loop_forward(final_combined,ocr_tokens) # (MAX_TOKENS)
		assert(copy_space.size()[0]==(self.MAX_TOKENS))

		# Final prediction
		prediction = F.softmax(torch.cat((answer_space,copy_space),dim=0),dim=1) # (VOCAB_DIM + MAX_TOKENS)
		assert(prediction.size()[0]==(self.VOCAB_DIM+self.MAX_TOKENS))

		return prediction

	def forward(self,text_fts:torch.tensor,img_fts:torch.tensor,obj_fts:torch.tensor,ocr_fts:torch.tensor,ocr_tokens:torch.tensor,num_tokens:torch.tensor):
		""" Takes all the contextual embeddings from the question, image and OCR to answer the question.
		
		Arguments :
			text_fts : (m,QUEST_DIM). Text features
			img_fts : (m,IMG_DIM). Image Grid features
			obj_fts : (m,IMG_DIM). Image Object features
			ocr_fts : (m,OCR_DIM). OCR Token features
			ocr_tokens : (m,MAX_TOKENS,EMBED_DIM)
			num_tokens : (m)
		"""

		BATCH_SIZE = text_fts.size()[0]

		# Combine Question and Grid Features
		out1 = self.net1(text_fts) # (m,IMG_DIM)
		assert(out1.size()==(BATCH_SIZE,self.IMG_DIM))
		question_grid = F.sigmoid(out1)*img_fts # (m,IMG_DIM)
		assert(question_grid.size()==(BATCH_SIZE,self.IMG_DIM))

		# Combine Question and Object Features
		out2 = self.net1(text_fts) # (m,IMG_DIM)
		assert(out2.size()==(BATCH_SIZE,self.IMG_DIM))
		question_obj = F.sigmoid(out2)*obj_fts # (m,IMG_DIM)
		assert(question_obj.size()==(BATCH_SIZE,self.IMG_DIM))

		# Combine Question and OCR Features
		out3 = self.net3(text_fts) # (m,IMG_DIM)
		assert(out3.size()==(BATCH_SIZE,self.IMG_DIM))
		out4 = self.net4(ocr_fts) # (m,IMG_DIM)
		assert(out4.size()==(BATCH_SIZE,self.IMG_DIM)), "Out 4 Size : {}".format(out4.size())
		question_ocr = F.sigmoid(out3)*out4 # (m,IMG_DIM)
		assert(question_ocr.size()==(BATCH_SIZE,self.IMG_DIM))

		# Addition of all 3 vectors
		final_combined = question_grid + question_obj + question_ocr #(m,IMG_DIM)

		# Linear projection to answer space
		answer_space = self.net5(final_combined) #(M,VOCAB_DIM)
		assert(answer_space.size()==(BATCH_SIZE,self.VOCAB_DIM))

		# Output of copy net
		copy_space = self.copy_net.forward(final_combined,ocr_tokens,num_tokens) #(M,MAX_TOKENS)
		assert(copy_space.size()==(BATCH_SIZE,self.MAX_TOKENS))

		# Final prediction
		prediction = F.softmax(torch.cat((answer_space,copy_space),dim=1),dim=1) #(M,VOCAB_DIM + MAX_TOKENS)
		assert(prediction.size()==(BATCH_SIZE,self.VOCAB_DIM+self.MAX_TOKENS))

		return prediction

if __name__ == '__main__' :

	import numpy as np

	MAX_TOKENS = 5
	BATCH_SIZE = 3
	IMG_DIM = 2048
	QUEST_DIM = 768
	OCR_DIM = 300
	EMBED_DIM = 300
	VOCAB_DIM = 10000

	text_inp = torch.randn((BATCH_SIZE,QUEST_DIM))
	ocr_inp = torch.randn((BATCH_SIZE,OCR_DIM))
	img_inp = torch.randn((BATCH_SIZE,IMG_DIM))
	obj_inp = torch.randn((BATCH_SIZE,IMG_DIM))
	raw_ocr_inp = [ torch.randn((2,EMBED_DIM)), torch.randn((4,EMBED_DIM)), torch.randn((3,EMBED_DIM))  ]
	num_tokens = torch.tensor([2,4,3])

	alex = CombineModes(QUEST_DIM,OCR_DIM,IMG_DIM,VOCAB_DIM,EMBED_DIM,MAX_TOKENS)

	# loop_forward(self,text_fts:torch.tensor,img_fts:torch.tensor,obj_fts:torch.tensor,ocr_fts:torch.tensor,ocr_tokens:torch.tensor)

	ans_list = []
	for i,token in zip(range(3),raw_ocr_inp) :
		ans_list.append( alex.loop_forward(text_inp[i],img_inp[i],obj_inp[i],ocr_inp[i],token).detach().cpu().numpy() )
	ans_list = np.array(ans_list)

	loop_final_ans = torch.tensor(ans_list)

	new_ocr_inp = torch.zeros( (BATCH_SIZE,MAX_TOKENS,EMBED_DIM) )
	new_ocr_inp[0,:2,:] = raw_ocr_inp[0]
	new_ocr_inp[1,:4,:] = raw_ocr_inp[1]
	new_ocr_inp[2,:3,:] = raw_ocr_inp[2]

	#text_fts:torch.tensor,img_fts:torch.tensor,obj_fts:torch.tensor,ocr_fts:torch.tensor,ocr_tokens:torch.tensor
	vectorized_final_ans = alex.forward(text_inp,img_inp,obj_inp,ocr_inp,new_ocr_inp,num_tokens)

	print(loop_final_ans[0,-5:])
	print(vectorized_final_ans[0,-5:])
	print()
	print()
	print(loop_final_ans[1,-5:])
	print(vectorized_final_ans[1,-5:])
	print()
	print()
	print(loop_final_ans[2,-5:])
	print(vectorized_final_ans[2,-5:])

	if torch.all(torch.eq(vectorized_final_ans,loop_final_ans)) :
		print("Success")
	else :
		eles = vectorized_final_ans != loop_final_ans

		print("vector eles : {}".format(vectorized_final_ans[eles]))
		print("loop eles : {}".format(loop_final_ans[eles]))