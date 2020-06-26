import torch, torch.nn as nn, random
import torchvision, torchvision.transforms as transforms, torchvision.models as models
import sys, os, pickle, matplotlib.pyplot as plt, numpy as np, PIL.Image as Image
from torch.autograd import Variable
from image_features.object_detection import NaiveObjectDetector
from utils.customDatasets import CustomDataset
from collections import defaultdict
from typing import List

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
        #if torch.cuda.is_available(): transform = transform.cuda()
        extracted_features = self.feature_module(transform).squeeze() 
        return extracted_features.detach()

class EndToEndFeatExtractor(nn.Module):
    def __init__(self, max_objects:int=50,feature = 'resnet152', object = 'fasterrcnn'):
        super().__init__()
        self.MAX_OBJECTS = max_objects
        self.object_detector = NaiveObjectDetector(type = object)
        self.feature_extractor = NaiveObjectFeatureExtractor(model = feature)
    
    def forward(self, image_paths:List[str],object_threshold = 0.4):
        """
        image argument is image path and never an image itself, 
        the object detector reads the image  
        """

        print("Image Paths : {}".format(image_paths))

        try: 
            boxes_list, class_list, masks_list, image_list = self.object_detector(image_paths, object_threshold)
            print("Success")
        except: 
            print("Exception")
            boxes_list, class_list, masks_list, image_list = self.object_detector(image_paths, 0.075)
            print("Very few prominent objects in the image")

        print("Num Boxes : {}".format(len(boxes_list)))

        num_boxes = []
        all_object_features = []
        for boxes,classes,masks,image in zip(boxes_list, class_list, masks_list, image_list):
            object_features = []
            print("Boxes retrieved")
            for k, box in enumerate(boxes):
                start_point, end_point = box[0], box[1]
                object_box = image[:, int(start_point[1]):int(end_point[1]), int(start_point[0]):int(end_point[0])]
                object_features.append(self.feature_extractor(object_box))
                print("\tbox {} processed".format(k + 1))
            print("All boxes processed")

            num_boxes.append( min(len(boxes),self.MAX_OBJECTS) )
            print("Old Length : {}".format(len(object_features)))
            if len(boxes) > self.MAX_OBJECTS:
                object_features = object_features[:-(len(boxes)-self.MAX_OBJECTS)]
            else :
                for k in range(self.MAX_OBJECTS-len(boxes)):
                    object_features.append(torch.zeros((2048)))
                    #object_features[-1] = object_features[-1].cuda()
            print("New Length : {}\n".format(len(object_features)))

            object_features = torch.stack(object_features)
            #object_features = object_features.unsqueeze(0)
            all_object_features.append(object_features)

        out_tensor = torch.stack(all_object_features)
        num_boxes = torch.tensor(num_boxes)
        return out_tensor,num_boxes

if __name__ == '__main__':
    # Use the dataloader to get a random image 
    data_path = 'Data'
    save_dir = 'inference_examples'
    os.makedirs(save_dir, exist_ok = True)
    mode = "train"
    ID_path = os.path.join(data_path, "{}/{}_ids.txt".format(mode,mode))
    json_path = os.path.join(data_path, "{}/cleaned.json".format(mode))
    tokens_path = os.path.join(data_path,"tokens_in_images.txt")
    dataloader = CustomDataset(data_path, ID_path, json_path, tokens_path, (448, 448), set_=mode)


    # get_path method has been described below, add method to CustomDataset
    inference_example1 = dataloader.get_path(random.randint(0, 10)) # Any image path to infer model on 
    inference_example2 = dataloader.get_path(random.randint(0, 20)) # Any image path to infer model on 
    inference_example3 = dataloader.get_path(random.randint(0, 20)) # Any image path to infer model on 
    inference_example4 = dataloader.get_path(random.randint(0, 20)) # Any image path to infer model on 
    inference_example5 = dataloader.get_path(random.randint(0, 20)) # Any image path to infer model on 
    # The variable inference example is an image path which is STR type
    
    print("\n--------------------------------------------To seperate the code output from the warnings --------------------------------------------\n")
    
    max_objects = 20
    end_to_end_feature_extractor = EndToEndFeatExtractor(max_objects)
    if torch.cuda.is_available(): end_to_end_feature_extractor = end_to_end_feature_extractor.cuda()
    
    out_tensor,num_boxes = end_to_end_feature_extractor([inference_example1,inference_example2,inference_example3,inference_example4,inference_example5],max_objects)
    print("Output tensor shape is {}".format(out_tensor.shape))
    print("Num Objects : {}".format(num_boxes))

    for idx,num in enumerate(num_boxes):
        base = idx*max_objects
        next_base = (idx+1)*max_objects
        assert( torch.all(torch.eq(out_tensor[base+num:next_base],torch.zeros((max_objects-num,2048)).cuda())) )

    # out_tensor will have shape (m, 2048), where m is the number of objects in that image




