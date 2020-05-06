import sys, os, time, argparse, cv2, numpy as np, json, zipfile, random, requests, matplotlib.pyplot as plt, collections, subprocess, math
import torch, torch.nn as nn, torch.nn.functional as functional, torch.backends.cudnn as cudnn, torch.nn.init as init
import torch, torch.nn as nn, torchvision.transforms as transforms
from IPython.display import Image as ipy_img
from google.colab.patches import cv2_imshow
from google.colab import drive
from torch.autograd import Variable
from PIL import Image
from skimage import io
from torch.autograd import Variable
from torchvision.models.vgg import model_urls
from torchvision import models
from collections import namedtuple
from collections import OrderedDict
from models import *
from utils import *

class TextDetection(nn.Module):
    def __init__(self, trained_model = '/content/drive/My Drive/TextVQA files/craft_mlt_25k.pth', refine = False, model = 'CRAFT', refiner_model = None, poly = False):
        super().__init__()
        self.refine = refine
        self.refiner_model = refiner_model
        self.poly = poly
        self.trained_model = trained_model
        if model == 'CRAFT':
            self.model = CRAFT()  
        
        if torch.cuda.is_available():
            self.model.load_state_dict(copyStateDict(torch.load(self.trained_model)))
            self.model = self.model.cuda()
            self.model = torch.nn.DataParallel(self.model)
            cudnn.benchmark = False
        else:
            self.model.load_state_dict(copyStateDict(torch.load(self.trained_model, map_location='cpu')))
        self.model.eval()

    def forward(self, image_path, text_threshold = 0.7, link_threshold = 0.4, low_text = 0.4, cuda = torch.cuda.is_available(), poly = False, canvas_size = 1280, mag_ratio = 1.5, refine_net=None):
        image = self._load_image(image_path)
        img_resized, target_ratio, size_heatmap = resize_aspect_ratio(image, canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=mag_ratio)
        ratio_h = ratio_w = 1 / target_ratio
        x = normalizeMeanVariance(img_resized)
        x = torch.from_numpy(x).permute(2, 0, 1)    
        x = Variable(x.unsqueeze(0))                
        if cuda:
            x = x.cuda()

        # forward pass
        with torch.no_grad():
            y, feature = self.model(x)

        # make score and link map
        score_text = y[0,:,:,0].cpu().data.numpy()
        score_link = y[0,:,:,1].cpu().data.numpy()

        # refine link
        if refine_net is not None:
            with torch.no_grad():
                y_refiner = refine_net(y, feature)
            score_link = y_refiner[0,:,:,0].cpu().data.numpy()

        # Post-processing
        boxes, polys = getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)

        # coordinate adjustment
        boxes = adjustResultCoordinates(boxes, ratio_w, ratio_h)
        polys = adjustResultCoordinates(polys, ratio_w, ratio_h)
        for k in range(len(polys)):
            if polys[k] is None: polys[k] = boxes[k]

        # render results (optional)
        render_img = score_text.copy()
        ret_score_text = cvt2HeatmapImg(render_img)
        return boxes, polys, ret_score_text

    def _load_image(self, image_path):
        return loadImage(image_path)

    def display(self, image_path, result_folder = './'):
        image = self._load_image(image_path)
        _, polys, _ = self.forward(image_path)
        ret_img = box_display(image_path, image[:,:,::-1], polys, dirname=result_folder)

    def write_heatmap(self, image_path, result_folder):
        filename, file_ext = os.path.splitext(os.path.basename(image_path))
        mask_file = result_folder + "/letter_heatmap_" + filename + '.jpg'
        _, _, score_text = self.forward(image_path)
        cv2.imwrite(mask_file, score_text)      

class TextRecognition(nn.Module):
    def __init__(self, model_path = "/content/drive/My Drive/TextVQA files/crnn.pth", alphabet = '0123456789abcdefghijklmnopqrstuvwxyz', model = 'CRNN'):
        super().__init__()
        if model == 'CRNN':
            self.model = CRNN(32, 1, 37, 256)
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        self.model.load_state_dict(torch.load(model_path))
        self.converter = strLabelConverter(alphabet)
        self.transformer = resizeNormalize((100, 32))
        self.model.eval()

    def forward(self, image):
        image = Image.fromarray(image)
        image = image.convert('L')
        image = self.transformer(image)
        if torch.cuda.is_available():
            image = image.cuda()
        image = image.view(1, *image.size())
        image = Variable(image)
        predictions = self.model(image)
        _, predictions = predictions.max(2)
        predictions = predictions.transpose(1, 0).contiguous().view(-1)
        preds_size = Variable(torch.IntTensor([predictions.size(0)]))
        raw_pred = self.converter.decode(predictions.data, preds_size.data, raw=True)
        sim_pred = self.converter.decode(predictions.data, preds_size.data, raw=False)
        return raw_pred, sim_pred

class EndToEndOCR(nn.Module):
    def __init__(self):
        super().__init__()
        self.text_detector = TextDetection()
        self.text_recognizer = TextRecognition()
        if torch.cuda.is_available():
            self.text_detector.cuda(), self.text_recognizer.cuda()
    
    def forward(self, image_path):
        ocr_tokens = list()
        if type(image_path) == str:
            image = cv2.imread(image_path)
        else:
            image = image_path  
        boxes, _, _ = self.text_detector(image_path)
        boxes = boxes.astype(np.int32)
        for box in boxes:
            x_max, x_min = max(box[:, 0]), min(box[:, 0])
            y_max, y_min = max(box[:, 1]), min(box[:, 1])
            tmp = image[y_min:y_max, x_min:x_max]
            _, sim_pred = self.text_recognizer(tmp)
            ocr_tokens.append(sim_pred)
        return ocr_tokens
