import sys, os ,cv2, random, matplotlib.pyplot as plt, numpy as np
import torchvision, torchvision.transforms as T, torch.nn as nn
from tqdm import tqdm
from PIL import Image
from utils.customDatasets import CustomDataset

class ObjectDetector(nn.Module):
    def __init__(self, type = 'fasterrcnn', labels_path = 'modules/labels.txt', num_classes = 91):
        super().__init__()
        self.type = type
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained = True, num_classes = num_classes)
        if self.type == 'maskrcnn': 
            self.model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained = True, num_classes = num_classes)
        self.model.eval()
        def get_classes(num_classes, file):
            lab_file, lab_list = open(file), []
            for label in lab_file.read().splitlines():
                lab_list.append(label.rsplit(": ")[1])
            return lab_list[:num_classes]
        self.classes = get_classes(num_classes, labels_path)

    def forward(self, image_path, threshold):
        img = Image.open(image_path)
        transform = T.Compose([T.ToTensor()])
        img = transform(img)
        pred = self.model([img])
        pred_class = [self.classes[i] for i in list(pred[0]['labels'].numpy())]
        pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().numpy())] 
        pred_score = list(pred[0]['scores'].detach().numpy())
        pred_t = [pred_score.index(x) for x in pred_score if x > threshold][-1] 
        pred_boxes = pred_boxes[:pred_t+1]
        pred_class = pred_class[:pred_t+1]
        pred_masks = None
        if self.type == 'maskrcnn':
            pred_masks = pred[0]['masks'].detach().numpy()
            pred_masks = pred_masks[:pred_t+1]
        return pred_boxes, pred_class, pred_masks

    def display_objects(self, image_path, boxes, class_label, masks, show_label = True, threshold = 0.5, rect_th = 2, text_size = 1, text_th = 2):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if masks is not None: 
            all_masks = np.zeros_like((np.vstack([masks[0]] * 3)).transpose(1, 2, 0))
            for i, mask in enumerate(masks):
                mask = (np.vstack([mask] * 3)).transpose(1, 2, 0) * 255
                all_masks += np.uint8(mask)
        for i in range(len(boxes)):
            cv2.rectangle(image, boxes[i][0], boxes[i][1],color=(255, 0, 0), thickness=rect_th)
            if show_label:
                cv2.putText(image,class_label[i], boxes[i][0],  cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,255,0),thickness=text_th)
        plt.figure(figsize = (20, 30))
        if masks is not None:
            plt.imshow(np.hstack([image / 255.0, all_masks]))
        else:
            plt.imshow(image)
        plt.xticks([]), plt.yticks([]), plt.show()
    
    def save_boxed_image(self, image_path, save_path, boxes, class_label, masks, show_label = True, threshold = 0.5, rect_th = 2, text_size = 1, text_th = 2):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if masks is not None: 
            all_masks = np.zeros_like((np.vstack([masks[0]] * 3)).transpose(1, 2, 0))
            for i, mask in enumerate(masks):
                mask = (np.vstack([mask] * 3)).transpose(1, 2, 0) * 255
                all_masks += np.uint8(mask)
        for i in range(len(boxes)):
            cv2.rectangle(image, boxes[i][0], boxes[i][1],color=(0, 255, 0), thickness=rect_th)
            if show_label:
                cv2.putText(image,class_label[i], boxes[i][0],  cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,255,0),thickness=text_th)
        if masks is not None:
            cv2.imwrite(os.path.join(save_path, image_path.rsplit('/')[-1]), cv2.cvtColor(np.hstack([image, all_masks]), cv2.COLOR_RGB2BGR))
        else:
            cv2.imwrite(os.path.join(save_path, image_path.rsplit('/')[-1]), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        print("Image saved as {}".format(str(os.path.join(save_path, image_path.rsplit('/')[-1]))))

if __name__ == '__main__':
    data_path = '/home/amanshenoy/Text_VQA/Data'
    ID_path = os.path.join(data_path, "test/test_ids.txt")
    json_path = os.path.join(data_path, "test/cleaned.json")
    dataloader = CustomDataset(data_path, ID_path, json_path, (448, 448), set_="test")

    # get_path method has been described below, add method to CustomDataset
    inference_example = dataloader.get_path(random.randint(0, 2048)) # Any image path to infer model on
    object_detector = ObjectDetector(type = 'fasterrcnn')

    # threshold (0.4) is the confidence value above which to consider a box relevant ; boxes and class_label are as expected
    boxes, class_label, masks = object_detector(inference_example, 0.4)
    object_detector.display_objects(inference_example, boxes, class_label, masks, show_label = True)
    save_dir = 'inference_examples'
    os.makedirs(save_dir, exist_ok = True)
    object_detector.save_boxed_image(inference_example, save_dir, boxes, class_label, masks, show_label = True)