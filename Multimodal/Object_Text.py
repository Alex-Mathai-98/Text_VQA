import torch
import torch.nn as nn
import torch.nn.functional as F

class ObjectFeaturesAndText(nn.Module):
	""" Combine Visual and Text Features """

	def __init__(self,embed_dim:int,img_ft_dims:list,MAX_OBJECTS:int) -> None:
		"""
		Arguments :
			embed_dim : The dimensionality for the question embedding
						ex. 768 for BERT
			img_ft_dims : The dimensionality of the object features from the feature extractor
						ex. 2048 for R-CNN 
		"""
		super(ObjectFeaturesAndText,self).__init__()

		self.embed_dim = embed_dim
		self.img_dim = img_ft_dims[0]
		self.MAX_OBJECTS = MAX_OBJECTS
		self.net1 = nn.Linear(embed_dim,self.img_dim)


	def loop_forward(self,text_fts:torch.tensor,img_fts:torch.tensor) -> torch.tensor:

		# img_fts : (m,img_dim)
		# text_fts : (embed_dim)

		#print("\n\n")
		#print("Text Size : {}".format(text_fts.size()))

		# out1 : (img_dim)
		out1 = self.net1(text_fts)
		#print(out1.size())

		# out1 : (m,img_dim)
		out1 = out1*img_fts
		#print(out1.size())

		# out1 : (m)
		out1 = torch.sum(out1,dim=1)
		#print(out1.size())

		# weights : (img_dim)
		weights = F.softmax(out1,dim=0).unsqueeze(1)
		#print("Weights : {}".format(weights.size()))

		# ans : (m,img_dim)
		ans = weights*img_fts
		#print("Ans : {}".format(ans.size()))

		# ans : (img_dim)
		ans = torch.sum(ans,dim=0)
		#print("Ans : {}".format(ans.size()))

		return ans

	def forward(self,text_fts:torch.tensor,img_fts:torch.tensor,num_objects:torch.tensor) -> torch.tensor:

		# text_fts : (m,embed_dim)
		# img_fts : (m,MAX_OBJECTS,img_dim)
		# attention_mask : (m)

		attention_mask = torch.arange(self.MAX_OBJECTS)[None, :] < num_objects[:, None]

		# out1 : (m,1,img_dim)
		out1 = self.net1(text_fts).unsqueeze(1)
		
		# out1 : (m,MAX_OBJECTS,img_dim)
		out1 = out1*img_fts
		
		# out1 : (m,MAX_OBJECTS)
		out1 = torch.sum(out1,dim=2)

		# hard coding the padded stuff to -inf
		out1[attention_mask==0] = -1*float("Inf")

		# out1 : (m,MAX_OBJECTS,1)
		out1 = F.softmax(out1,dim=1).unsqueeze(2)

		# ans : (m,MAX_OBJECTS,img_dim)
		ans = out1*img_fts

		# ans : (m,img_dim)
		ans = torch.sum(ans,dim=1)

		return ans


if __name__ == '__main__' :

	import numpy as np

	MAX_OBJECTS = 5
	batch_size = 3
	IMG_DIM = 2048
	EMBED_DIM = 768

	text_inp = torch.randn((3,EMBED_DIM))
	img_inp = [ torch.randn((2,IMG_DIM)), torch.randn((4,IMG_DIM)), torch.randn((3,IMG_DIM))  ]
	num_objects = torch.tensor([2,4,3])

	alex = ObjectFeaturesAndText(768,[IMG_DIM],MAX_OBJECTS)

	ans_list = []
	for i,img in zip(range(3),img_inp) :
		ans_list.append( alex.forward(text_inp[i],img).detach().cpu().numpy() )
	ans_list = np.array(ans_list)

	loop_final_ans = torch.tensor(ans_list)

	new_img_inp = torch.zeros( (batch_size,MAX_OBJECTS,IMG_DIM) )
	new_img_inp[0,:2,:] = img_inp[0]
	new_img_inp[1,:4,:] = img_inp[1]
	new_img_inp[2,:3,:] = img_inp[2]

	vectorized_final_ans = alex.vectorized_forward(text_inp,new_img_inp,num_objects)

	print(loop_final_ans[0])
	print(vectorized_final_ans[0])

	if torch.all(torch.eq(vectorized_final_ans,loop_final_ans)) :
		print("Success")
	else :
		eles = vectorized_final_ans != loop_final_ans

		print("vector eles : {}".format(vectorized_final_ans[eles]))
		print("loop eles : {}".format(loop_final_ans[eles]))