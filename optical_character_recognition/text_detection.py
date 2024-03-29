import sys, os, time, argparse, cv2, numpy as np, random
import torch, torch.nn as nn, torch.nn.functional as functional
from optical_character_recognition import utils
from torch.autograd import Variable
from optical_character_recognition.models import CRAFT
from optical_character_recognition.models import RefineNet
from optical_character_recognition.text_recognition import TextRecognition

class TextDetection(nn.Module):
    def __init__(self, trained_model = 'optical_character_recognition/pretrained_models/craft_mlt_25k.pth', refiner_model = 'optical_character_recognition/pretrained_models/craft_refiner_CTW1500.pth'):
        """
        Text Detection for boxing the text in any given image 
        """
        super().__init__()
        self.refiner_model = refiner_model
        self.trained_model = trained_model
        self.model = CRAFT()  
        
        if torch.cuda.is_available():
            self.model.load_state_dict(utils.copyStateDict(torch.load(self.trained_model)))
            self.model = self.model.cuda()
        else: self.model.load_state_dict(utils.copyStateDict(torch.load(self.trained_model, map_location='cpu')))
        
        self.model.eval()
        self.refine_net = RefineNet()

        if torch.cuda.is_available():
            self.refine_net.load_state_dict(utils.copyStateDict(torch.load(self.refiner_model)))
            self.refine_net = self.refine_net.cuda()
        else: self.refine_net.load_state_dict(utils.copyStateDict(torch.load(self.refiner_model, map_location='cpu')))
        
        self.refine_net.eval()
    
    def forward(self, image_path, text_threshold = 0.7, link_threshold = 0.4, low_text = 0.4, canvas_size = 1280, mag_ratio = 1.5, refine = True):
        """
        Given image path return boxes/polys for refined or unrefined model. Most the arguments are self explanatory, 
        refine is the flag to use phrases (refine = True) or not (refine = False)   
        """
        # Load image if required 
        if type(image_path) == str:
            image = utils.loadImage(image_path)
        elif type(image_path) == np.ndarray:
            image = image_path

        # Typical pre-processing - normalize, resize, convert to batch
        img_resized, target_ratio, size_heatmap = utils.resize_aspect_ratio(image, canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=mag_ratio)
        ratio_h = ratio_w = 1 / target_ratio
        x = utils.normalizeMeanVariance(img_resized)
        x = torch.from_numpy(x).permute(2, 0, 1)    
        x = Variable(x.unsqueeze(0))                
        
        if torch.cuda.is_available(): x = x.cuda()

        # forward pass
        with torch.no_grad():
            y, feature = self.model(x)

        # make region and affinity score map
        region_score = y[0,:,:,0].cpu().data.numpy()
        affinity_score = y[0,:,:,1].cpu().data.numpy()
        extrapolated_region_score = cv2.resize(region_score, image.shape[:2][::-1], interpolation = cv2.INTER_CUBIC)
        extrapolated_affinity_score = cv2.resize(affinity_score, image.shape[:2][::-1], interpolation = cv2.INTER_CUBIC)

        # refine link
        if refine:
            with torch.no_grad():
                y_refiner = self.refine_net(y, feature)
            affinity_score = y_refiner[0,:,:,0].cpu().data.numpy()
        
        # Post-processing
        boxes, polys = utils.getDetBoxes(region_score, affinity_score, text_threshold, link_threshold, low_text, True)
        #refined_boxes, refined_polys = utils.getDetBoxes(region_score, refiner_affinity_score, text_threshold, link_threshold, low_text, True)

        # coordinate adjustment
        boxes = utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
        polys = utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
        #refined_boxes = utils.adjustResultCoordinates(refined_boxes, ratio_w, ratio_h)
        #refined_polys = utils.adjustResultCoordinates(refined_polys, ratio_w, ratio_h)
        
        # If not a polygon, just use the box 
        for k in range(len(polys)):
            if polys[k] is None: polys[k] = boxes[k]

        return boxes, polys, extrapolated_region_score, extrapolated_affinity_score 

    def display(self, image_path, result_folder = '.', show = True, threshold = 0.9, refine = True):
        """
        Display the image with all the text boxes for verification 
        """
        image = utils.loadImage(image_path)
        boxes, polys, _, _ = self.forward(image_path, text_threshold=threshold, refine = refine)
        image = utils.display(image_path, image[:,:,::-1], polys, dirname=result_folder, show = show)
        return image, boxes, polys   

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='CRAFT Text Detection')

    parser.add_argument('--test_folder', default='Data/test/test_images/', type=str, help='folder path to input images')
    parser.add_argument('--test_image', default=None, type=str, help='To test an image')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result_folder = './optical_character_recognition/inference_examples/'

    td = TextDetection().to(device)
    td.display(args.test_image)