import torch
import torch.nn as nn
import torch.nn.functional as F

class ObjectFeaturesAndText(nn.Module):
	""" Combine Visual and Text Features """

	def __init__(self,QUEST_DIM:int,img_ft_dims:list,MAX_OBJECTS:int) -> None:
		"""
		Arguments :
			QUEST_DIM : The dimensionality for the question embedding
						ex. 768 for BERT
			img_ft_dims : The dimensionality of the object features from the feature extractor
						ex. 2048 for R-CNN 
		"""
		super(ObjectFeaturesAndText,self).__init__()

		self.QUEST_DIM = QUEST_DIM
		self.IMG_DIM = img_ft_dims[0]
		self.MAX_OBJECTS = MAX_OBJECTS
		self.net1 = nn.Linear(self.QUEST_DIM,self.IMG_DIM)


	def loop_forward(self,text_fts:torch.tensor,img_fts:torch.tensor) -> torch.tensor:

		# img_fts : (m,IMG_DIM)
		# text_fts : (QUEST_DIM)

		#print("\n\n")
		#print("Text Size : {}".format(text_fts.size()))

		# out1 : (IMG_DIM)
		out1 = self.net1(text_fts)
		#print(out1.size())

		# out1 : (m,IMG_DIM)
		out1 = out1*img_fts
		#print(out1.size())

		# out1 : (m)
		out1 = torch.sum(out1,dim=1)
		#print(out1.size())

		# weights : (IMG_DIM)
		weights = F.softmax(out1,dim=0).unsqueeze(1)
		#print("Weights : {}".format(weights.size()))

		# ans : (m,IMG_DIM)
		ans = weights*img_fts
		#print("Ans : {}".format(ans.size()))

		# ans : (IMG_DIM)
		ans = torch.sum(ans,dim=0)
		#print("Ans : {}".format(ans.size()))

		return ans

	def forward(self,text_fts:torch.tensor,img_fts:torch.tensor,num_objects:torch.tensor) -> torch.tensor:

		# text_fts : (m,QUEST_DIM)
		# img_fts : (m,MAX_OBJECTS,IMG_DIM)
		# attention_mask : (m)

		BATCH_SIZE = text_fts.size()[0]

		attention_mask = torch.arange(self.MAX_OBJECTS)[None, :] < num_objects[:, None]

		# out1 : (m,1,IMG_DIM)
		out1 = self.net1(text_fts).unsqueeze(1)
		assert(out1.size() == (BATCH_SIZE,1,self.IMG_DIM))

		# out1 : (m,MAX_OBJECTS,IMG_DIM)
		out1 = out1*img_fts
		assert(out1.size() == (BATCH_SIZE,self.MAX_OBJECTS,self.IMG_DIM))

		# out1 : (m,MAX_OBJECTS)
		out1 = torch.sum(out1,dim=2)
		assert(out1.size()==(BATCH_SIZE,self.MAX_OBJECTS))

		# hard coding the padded stuff to -inf
		out1[attention_mask==0] = -1*float("Inf")

		# out1 : (m,MAX_OBJECTS,1)
		out1 = F.softmax(out1,dim=1).unsqueeze(2)
		assert(out1.size()==(BATCH_SIZE,self.MAX_OBJECTS,1))

		# ans : (m,MAX_OBJECTS,IMG_DIM)
		ans = out1*img_fts
		assert(ans.size()==(BATCH_SIZE,self.MAX_OBJECTS,self.IMG_DIM))

		# ans : (m,IMG_DIM)
		ans = torch.sum(ans,dim=1)
		assert(ans.size()==(BATCH_SIZE,self.IMG_DIM))

		return ans


if __name__ == '__main__' :

	import numpy as np

	MAX_OBJECTS = 5
	batch_size = 3
	IMG_DIM = 2048
	QUEST_DIM = 768

	text_inp = torch.randn((3,QUEST_DIM))
	img_inp = [ torch.randn((2,IMG_DIM)), torch.randn((4,IMG_DIM)), torch.randn((3,IMG_DIM))  ]
	num_objects = torch.tensor([2,4,3])

	alex = ObjectFeaturesAndText(768,[IMG_DIM],MAX_OBJECTS)

	ans_list = []
	for i,img in zip(range(3),img_inp) :
		ans_list.append( alex.loop_forward(text_inp[i],img).detach().cpu().numpy() )
	ans_list = np.array(ans_list)

	loop_final_ans = torch.tensor(ans_list)

	new_img_inp = torch.zeros( (batch_size,MAX_OBJECTS,IMG_DIM) )
	new_img_inp[0,:2,:] = img_inp[0]
	new_img_inp[1,:4,:] = img_inp[1]
	new_img_inp[2,:3,:] = img_inp[2]

	vectorized_final_ans = alex.forward(text_inp,new_img_inp,num_objects)

	print(loop_final_ans[0])
	print(vectorized_final_ans[0])

	if torch.all(torch.eq(vectorized_final_ans,loop_final_ans)) :
		print("Success")
	else :
		eles = vectorized_final_ans != loop_final_ans

		print("vector eles : {}".format(vectorized_final_ans[eles]))
		print("loop eles : {}".format(loop_final_ans[eles]))