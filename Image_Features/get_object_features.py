import sys, os ,cv2, random, matplotlib.pyplot as plt, numpy as np
import torchvision, torchvision.transforms as transforms, torch.nn as nn
import torch, pickle
import torchvision.models as models
from tqdm import tqdm
from PIL import Image
from Image_Features.object_detection import ObjectDetector
from utils.customDatasets import CustomDataset
from torch.autograd import Variable

TARGET_IMAGE_SIZE = [448, 448]
CHANNEL_MEAN = [0.485, 0.456, 0.406]
CHANNEL_STD = [0.229, 0.224, 0.225]
data_transforms = transforms.Compose([
        transforms.Resize(TARGET_IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(CHANNEL_MEAN, CHANNEL_STD),
    ])

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

    def forward(self, x):
        transform = data_transforms(x)
        if transform.shape[0] == 1:
            transform = transform.expand(3, -1, -1)
        transform = Variable(transform.unsqueeze(0))
        if torch.cuda.is_available():
            transform = transform.cuda()
        return self.feature_module(transform).squeeze()

class FinetuneFasterRcnnFpnFc7(nn.Module):
    def __init__(self, in_dim, weights_file, bias_file, model_data_dir):
        super(FinetuneFasterRcnnFpnFc7, self).__init__()
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

path = 'Data/test/test_images/c36e1c503400ee40.jpg'
img = Image.open(path).convert("RGB")
obj, finetuner = ObjectFeatureExtractor(), FinetuneFasterRcnnFpnFc7(2048, 'detectron/fc6/fc7_w.pkl', 'detectron/fc6/fc7_b.pkl', '../pythia/data/').cuda()
feat = obj(img)
final_feat = finetuner(feat)
print(final_feat.shape)