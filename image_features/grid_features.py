from torchvision import models
import torch
from torch import nn

class GridFeatures(nn.Module) :

	def __init__(self,code=None):

		super(GridFeatures,self).__init__()

		if code == "resnet101" :
			resnet = models.resnet101(pretrained=True)
			self.model = torch.nn.Sequential( *list(resnet.children())[:-2] )

	def forward(self,X):
		return self.model(X)

if __name__ == '__main__' :

	X = torch.randn((1,3,448,448))
	alex = GridFeatures("resnet101")
	t1 = alex(X)
	print("Shape : {}".format(t1.size()))