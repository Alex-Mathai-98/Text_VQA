import os ,numpy as np, pickle
import torch, torch.nn as nn, torchvision
from tqdm import tqdm
from argparse import ArgumentParser
from collections import defaultdict
from utils.customDatasets import CustomDataset
from image_features.object_detection import ObjectDetector
from image_features.feature_extractor import ObjectFeatureExtractor

if __name__ == '__main__':
    parser = ArgumentParser(description = "Get object level features for all objects in all images in a given directory")
    parser.add_argument("--data_path", help = 'Directory containing all the images', default = 'Data')
    parser.add_argument("--save_path", help = 'Where to save object features in dictionary format', default = 'Data/test/object_level_features')
    parser.add_argument("--id_path", help = 'Directory containing all the images', default = "Data/test/test_ids.txt")
    parser.add_argument("--json_path", help = 'Where to save object features in dictionary format', default = "Data/test/cleaned.json")
    parser.add_argument("--mode", help = "Test / Dev / Train set", default = 'test')
    args = parser.parse_args()
    os.makedirs(args.save_path, exist_ok = True)
    dataloader = CustomDataset(args.data_path, args.id_path, args.json_path, (448, 448), set_= "test")
    object_detector, feature_extractor = ObjectDetector(), ObjectFeatureExtractor()

    for i in tqdm(range(dataloader.__len__())):
        if args.mode == 'test':
            feature_details = defaultdict()
            feature_details["image_id"], feature_details["path"] = dataloader.get_ID(i), dataloader.get_path(i)
            feature_details["boxes"], feature_details["classes"], feature_details["masks"] = object_detector(feature_details["path"])
            feature_details["image"] = object_detector.get_image_from_path(feature_details["path"])
            feature_details["image_feats"], feature_details["objects"], feature_details["num_objects"] = [], [], len(feature_details["classes"])
            for box in feature_details["boxes"]:
                start_point, end_point = box[0], box[1]
                object_image = feature_details["image"][int(start_point[1]):int(end_point[1]), int(start_point[0]):int(end_point[0]),:]
                feature_details["objects"].append(object_image)
                feature_details["image_feats"].append(feature_extractor(object_image))
            with open(os.path.join(args.save_path, feature_details["image_id"] + '.pkl'), 'wb') as handle:
                pickle.dump(feature_details, handle, protocol=pickle.HIGHEST_PROTOCOL)