import sys, os, time, argparse, cv2, numpy as np, json, zipfile, random
import torch, torch.nn as nn, torch.nn.functional as functional, torch.backends.cudnn as cudnn
from optical_character_recognition import utils
from torch.autograd import Variable
from PIL import Image
from skimage import io
from optical_character_recognition.models import CRAFT
from collections import OrderedDict

def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")

def text_detector(net, image, text_threshold = 0.7, link_threshold = 0.4, low_text = 0.4, cuda = torch.cuda.is_available(), poly = False, canvas_size = 1280, mag_ratio = 1.5, refine_net=None):
    img_resized, target_ratio, size_heatmap = utils.resize_aspect_ratio(image, canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=mag_ratio)
    ratio_h = ratio_w = 1 / target_ratio
    x = utils.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]
    if cuda:
        x = x.cuda()

    # forward pass
    with torch.no_grad():
        y, feature = net(x)

    # make score and link map
    score_text = y[0,:,:,0].cpu().data.numpy()
    score_link = y[0,:,:,1].cpu().data.numpy()

    # refine link
    if refine_net is not None:
        with torch.no_grad():
            y_refiner = refine_net(y, feature)
        score_link = y_refiner[0,:,:,0].cpu().data.numpy()

    # Post-processing
    boxes, polys = utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)

    # coordinate adjustment
    boxes = utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
    for k in range(len(polys)):
        if polys[k] is None: polys[k] = boxes[k]

    # render results (optional)
    render_img = score_text.copy()
    ret_score_text = utils.cvt2HeatmapImg(render_img)
    return boxes, polys, ret_score_text

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='CRAFT Text Detection')
    parser.add_argument('--trained_model', default='optical_character_recognition/pretrained_models/craft_mlt_25k.pth', type=str, help='pretrained model')
    parser.add_argument('--text_threshold', default=0.7, type=float, help='text confidence threshold')
    parser.add_argument('--low_text', default=0.4, type=float, help='text low-bound score')
    parser.add_argument('--link_threshold', default=0.4, type=float, help='link confidence threshold')
    parser.add_argument('--cuda', default=torch.cuda.is_available(), type=bool, help='Use cuda for inference')
    parser.add_argument('--canvas_size', default=1280, type=int, help='image size for inference')
    parser.add_argument('--mag_ratio', default=1.5, type=float, help='image magnification ratio')
    parser.add_argument('--poly', default=False, action='store_true', help='enable polygon type')
    parser.add_argument('--show_time', default=False, action='store_true', help='show processing time')
    parser.add_argument('--test_folder', default='Data/test/test_images/', type=str, help='folder path to input images')
    parser.add_argument('--refine', default=False, action='store_true', help='enable link refiner')
    parser.add_argument('--refiner_model', default='weights/craft_refiner_CTW1500.pth', type=str, help='pretrained refiner model')
    args = parser.parse_args()

    result_folder = './optical_character_recognition/inference_examples/'
    if not os.path.isdir(result_folder):
        os.mkdir(result_folder)

    t0 = time.time()
    net = CRAFT()
    if args.cuda:
        net.load_state_dict(copyStateDict(torch.load(args.trained_model)))
    else:
        net.load_state_dict(copyStateDict(torch.load(args.trained_model, map_location='cpu')))
    if args.cuda:
        net = net.cuda()
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = False
    net.eval()

    if args.refine:
        from refinenet import RefineNet
        refine_net = RefineNet()
        print('Loading weights of refiner from checkpoint (' + args.refiner_model + ')')
        if args.cuda:
            refine_net.load_state_dict(copyStateDict(torch.load(args.refiner_model)))
            refine_net = refine_net.cuda()
            refine_net = torch.nn.DataParallel(refine_net)
        else:
            refine_net.load_state_dict(copyStateDict(torch.load(args.refiner_model, map_location='cpu')))
        refine_net.eval()
        args.poly = True
    
    image_root = args.test_folder
    image_dir = os.listdir(image_root)
    image_path = os.path.join(image_root, random.choice(image_dir))
    image = utils.loadImage(image_path)
    bboxes, polys, score_text = text_detector(net, image, args.text_threshold, args.link_threshold, args.low_text, args.cuda, args.poly, args.canvas_size, args.mag_ratio, refine_net = None)
    filename, file_ext = os.path.splitext(os.path.basename(image_path))
    mask_file = result_folder + "/letter_heatmap_" + filename + '.jpg'
    cv2.imwrite(mask_file, score_text)
    utils.display(image_path, image[:,:,::-1], polys, dirname=result_folder)