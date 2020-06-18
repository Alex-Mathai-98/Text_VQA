import torch, torch.nn as nn, random
import torchvision, torchvision.transforms as transforms, torchvision.models as models
import sys, os ,cv2, pickle, matplotlib.pyplot as plt, numpy as np, PIL.Image as Image
from torch.autograd import Variable
from image_features.object_detection import NaiveObjectDetector
from utils.customDatasets import CustomDataset
from collections import defaultdict

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
        if torch.cuda.is_available(): transform = transform.cuda()
        extracted_features = self.feature_module(transform).squeeze() 
        return extracted_features.detach()

class EndToEndFeatExtractor(nn.Module):
    def __init__(self, feature = 'resnet152', object = 'fasterrcnn'):
        super().__init__()
        self.object_detector = NaiveObjectDetector(type = object)
        self.feature_extractor = NaiveObjectFeatureExtractor(model = feature)
    
    def forward(self, image, object_threshold = 0.4):
        """
        image argument is image path and never an image itself, 
        the object detector reads the image  
        """
        try: boxes, _, _, image = self.object_detector(inference_example, object_threshold)
        except: 
            boxes, _, _, image = self.object_detector(inference_example, 0.075)
            print("Very few prominent objects in the image")

        object_features = []
        print("Boxes retrieved")
        for k, box in enumerate(boxes):
            start_point, end_point = box[0], box[1]
            object_box = image[:, int(start_point[1]):int(end_point[1]), int(start_point[0]):int(end_point[0])]
            object_features.append(self.feature_extractor(object_box))
            print("box {} processed".format(k + 1))
        print("All boxes processed")
        out_tensor = torch.stack(object_features)
        return out_tensor

if __name__ == '__main__':
    # Use the dataloader to get a random image 
    data_path = 'Data'
    save_dir = 'inference_examples'
    os.makedirs(save_dir, exist_ok = True)
    ID_path = os.path.join(data_path, "test/test_ids.txt")
    json_path = os.path.join(data_path, "test/cleaned.json")
    dataloader = CustomDataset(data_path, ID_path, json_path, (448, 448), set_="test")

    # get_path method has been described below, add method to CustomDataset
    inference_example = dataloader.get_path(random.randint(0, 2048)) # Any image path to infer model on 
    # The variable inference example is an image path which is STR type
    
    print("\n--------------------------------------------To seperate the code output from the warnings --------------------------------------------\n")
    
    end_to_end_feature_extractor = EndToEndFeatExtractor()
    if torch.cuda.is_available(): end_to_end_feature_extractor = end_to_end_feature_extractor.cuda()
    out_tensor = end_to_end_feature_extractor(inference_example)
    print("Output tensor shape is {}".format(out_tensor.shape))
    # out_tensor will have shape (m, 2048), where m is the number of objects in that image




