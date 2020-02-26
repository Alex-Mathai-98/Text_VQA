import torch, torch.nn as nn
import torchvision, torchvision.transforms as transforms, torchvision.models as models
import sys, os ,cv2, pickle, matplotlib.pyplot as plt, numpy as np, PIL.Image as Image
from torch.autograd import Variable

TARGET_IMAGE_SIZE = [448, 448]
CHANNEL_MEAN = [0.485, 0.456, 0.406]
CHANNEL_STD = [0.229, 0.224, 0.225]
data_transforms = transforms.Compose([transforms.Resize(TARGET_IMAGE_SIZE),
                                      transforms.ToTensor(),
                                      transforms.Normalize(CHANNEL_MEAN, CHANNEL_STD)])

class FinetunedLinear(nn.Module):
    def __init__(self, in_dim, weights_file = 'fc7_w.pkl', bias_file = 'fc7_b.pkl', model_data_dir = 'Image_Features/pretrained_weights'):
        super().__init__()
        if not os.path.isabs(weights_file):
            weights_file = os.path.join(model_data_dir, weights_file)
        if not os.path.isabs(bias_file):
            bias_file = os.path.join(model_data_dir, bias_file)
        with open(weights_file, "rb") as w:
            weights = pickle.load(w)
        with open(bias_file, "rb") as b:
            bias = pickle.load(b)
        out_dim = bias.shape[0]

        self.lc = nn.Linear(in_dim, out_dim)
        self.lc.weight.data.copy_(torch.from_numpy(weights))
        self.lc.bias.data.copy_(torch.from_numpy(bias))
        self.out_dim = out_dim

    def forward(self, image):
        i2 = self.lc(image)
        i3 = nn.functional.relu(i2)
        return i3

class ObjectFeatureExtractor(nn.Module):
    def __init__(self, model = 'resnet152'):
        super().__init__()
        if model == 'resnet152':
            self.model = models.resnet152(pretrained = True)
        self.model.eval()
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        modules = list(self.model.children())[:-1]
        self.feature_module = nn.Sequential(*modules)
        self.finetuner = FinetunedLinear(2048)

    def forward(self, x):
        transform = data_transforms(x)
        if transform.shape[0] == 1:
            transform = transform.expand(3, -1, -1)
        transform = Variable(transform.unsqueeze(0))
        if torch.cuda.is_available():
            transform = transform.cuda()
        extracted_features = self.feature_module(transform).squeeze() 
        return self.finetuner(extracted_features)

if __name__ == '__main__':
    path = 'Data/test/test_images/c36e1c503400ee40.jpg'
    image = Image.open(path).convert("RGB")
    feat_extract = ObjectFeatureExtractor().cuda()
    image_feature = feat_extract(image)
    print(image_feature.shape)

