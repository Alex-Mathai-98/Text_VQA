import sys, os ,cv2, random, matplotlib.pyplot as plt, numpy as np, pickle
import torchvision, torchvision.transforms as T, torch.nn as nn, torch
from tqdm import tqdm
from PIL import Image
from utils.customDatasets import CustomDataset
from argparse import ArgumentParser

if __name__ == '__main__':
    parser = ArgumentParser(description = "Get object level features for all objects in all images in a given directory")
    parser.add_argument("--data_path", help = 'Directory containing all the images', default = 'Data')
    parser.add_argument("--save_path", help = 'Where to save object features in dictionary format', default = 'Data/test/object_level_features')
    parser.add_argument("--id_path", help = 'Directory containing all the images', default = "Data/test/test_ids.txt")
    parser.add_argument("--json_path", help = 'Where to save object features in dictionary format', default = "Data/test/cleaned.json")
    args = parser.parse_args()
    os.makedirs(args.save_path, exist_ok = True)
    dataloader = CustomDataset(args.data_path, args.id_path, args.json_path, (448, 448), set_= "test")

    for i in tqdm(range(dataloader.__len__())):
        pass


    """
    # get_path method has been described below, add method to CustomDataset
    inference_example = dataloader.get_path(random.randint(0, 2048)) # Any image path to infer model on   
    object_detector = ObjectDetector(type = 'fasterrcnn')

    # threshold (0.3) is the confidence value above which to consider a box relevant ; boxes and class_label are as expected
    boxes, class_label, masks = object_detector(inference_example, 0.3)
    object_detector.display_objects(inference_example, boxes, class_label, masks, show_label = True)
    object_detector.save_boxed_image(inference_example, save_dir, boxes, class_label, masks, show_label = True)
    """