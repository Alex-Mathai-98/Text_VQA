import os ,numpy as np, pickle, gc
import torch, torch.nn as nn, torchvision
from tqdm import tqdm 
from argparse import ArgumentParser
from collections import defaultdict
from utils.customDatasets import CustomDataset
from image_features.object_detection import NaiveObjectDetector
from image_features.feature_extractor import NaiveObjectFeatureExtractor

if __name__ == '__main__':
    parser = ArgumentParser(description = "Get object level features for all objects in all images in a given directory")
    parser.add_argument("--data_path", help = 'Directory containing all the images', default = 'Data')
    parser.add_argument("--save_path", help = 'Where to save object features in dictionary format', default = 'Data/test/object_level_features')
    parser.add_argument("--id_path", help = 'Directory containing all the images', default = "Data/test/test_ids.txt")
    parser.add_argument("--json_path", help = 'Where to save object features in dictionary format', default = "Data/test/cleaned.json")
    parser.add_argument("--mode", help = "Test / Dev / Train set", default = 'test')
    args = parser.parse_args()
    os.makedirs(args.save_path, exist_ok = True)
    dataloader = CustomDataset(args.data_path, args.id_path, args.json_path, (448, 448), set_= args.mode)
    object_detector, feature_extractor = NaiveObjectDetector(), NaiveObjectFeatureExtractor()
    if torch.cuda.is_available(): object_detector, feature_extractor = NaiveObjectDetector().cuda(), NaiveObjectFeatureExtractor().cuda(1)
   
    for  data in tqdm(dataloader):
        if args.mode == 'test':
            feature_details = defaultdict()
            feature_details["image"], feature_details["transformed_image"], feature_details["question"], feature_details["index"], _, _ = data
            feature_details["image_id"], feature_details["path"] = dataloader.get_ID(feature_details["index"]), dataloader.get_path(feature_details["index"])
            try: feature_details["boxes"], feature_details["classes"], feature_details["masks"] = object_detector(feature_details["image"], threshold = 0.2)
            except:
                feature_details["boxes"], feature_details["classes"], feature_details["masks"] = [], [], []
                print("NoObjectDetected: Either image {} does not have objects prevalent enough or threshold is too high".format(feature_details["image_id"]))
            feature_details["image_feats"], feature_details["num_objects"] = [], len(feature_details["classes"])    
            for box in feature_details["boxes"]:
                start_point, end_point = box[0], box[1]
                object_image = feature_details["image"][:, int(start_point[1]):int(end_point[1]), int(start_point[0]):int(end_point[0])]
                feature_details["image_feats"].append(feature_extractor(object_image))
            with open(os.path.join(args.save_path, feature_details["image_id"] + '.pkl'), 'wb') as handle:
                pickle.dump(feature_details, handle, protocol=pickle.HIGHEST_PROTOCOL)