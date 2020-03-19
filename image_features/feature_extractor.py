import torch, torch.nn as nn
import torchvision, torchvision.transforms as transforms, torchvision.models as models
import sys, os ,cv2, pickle, matplotlib.pyplot as plt, numpy as np, PIL.Image as Image
from torch.autograd import Variable

TARGET_IMAGE_SIZE = [448, 448]
CHANNEL_MEAN = [0.485, 0.456, 0.406]
CHANNEL_STD = [0.229, 0.224, 0.225]
data_transforms = transforms.Compose([transforms.ToPILImage(),
                                      transforms.Resize(TARGET_IMAGE_SIZE),
                                      transforms.ToTensor(),
                                      transforms.Normalize(CHANNEL_MEAN, CHANNEL_STD)])

class NaiveObjectFeatureExtractor(nn.Module):
    def __init__(self, model = 'resnet152'):
        super().__init__()
        if model == 'resnet152':
            self.model = models.resnet152(pretrained = True)
        self.model.eval()
        modules = list(self.model.children())[:-1]
        self.feature_module = nn.Sequential(*modules)

    def forward(self, img):
        img = data_transforms(img.cpu())
        if img.shape[0] == 1:
            img = img.expand(3, -1, -1)
        transform = Variable(img.unsqueeze(0))
        if torch.cuda.is_available(): transform = transform.cuda(1)
        extracted_features = self.feature_module(transform).squeeze() 
        return extracted_features.detach()