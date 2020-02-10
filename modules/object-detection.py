import sys, os ,cv2, matplotlib.pyplot as plt
import torchvision, torchvision.transforms as T, torch.nn as nn
from tqdm import tqdm
from PIL import Image
from utils.customLoader import CustomDataset

class ObjectDetector(nn.Module):
    def __init__(self, threshold=0.5, rect_th=3, text_size=3, text_th=3):
        super().__init__()
        self.threshold = threshold
        self.rect_th = rect_th
        self.text_size = text_size
        self.text_th = text_th
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained = True)
        self.model.eval()
        self.classes = ['__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                        'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
                        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
                        'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
                        'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                        'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
                        'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
                        'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                        'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
                        'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
                        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
                        'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

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
        return pred_boxes, pred_class

    def display_objects(self, image_path, boxes, class_label, show_label = True, threshold = 0.5, rect_th = 3, text_size = 3, text_th = 3):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        for i in range(len(boxes)):
            cv2.rectangle(image, boxes[i][0], boxes[i][1],color=(0, 255, 0), thickness=rect_th)
            if show_label:
                cv2.putText(image,class_label[i], boxes[i][0],  cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,255,0),thickness=text_th)
        plt.figure(figsize = (20, 30))
        plt.imshow(image)
        plt.xticks([])
        plt.yticks([])
        plt.show()
    
    def save_boxed_image(self, image_path, save_path, boxes, class_label, show_label = True, threshold = 0.5, rect_th = 3, text_size = 3, text_th = 3):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        for i in range(len(boxes)):
            cv2.rectangle(image, boxes[i][0], boxes[i][1],color=(0, 255, 0), thickness=rect_th)
            if show_label:
                cv2.putText(image,class_label[i], boxes[i][0],  cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,255,0),thickness=text_th)
        cv2.imwrite(os.path.join(save_path, image_path.rsplit('/')[-1]), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        print("Image saved as {}".format(str(os.path.join(save_path, image_path.rsplit('/')[-1]))))

if __name__ == '__main__':

    data_path = '/home/amanshenoy/Text_VQA/Data'
    ID_path = os.path.join(data_path, "test/test_ids.txt")
    json_path = os.path.join(data_path, "test/cleaned.json")
    dataloader = CustomDataset(data_path, ID_path, json_path, (448, 448), set_="test")
    # get_path method has been described below, add method to CustomDataset
    inference_example = dataloader.get_path(12)
    """
    def get_path(self, index):
        ID = self.list_IDs[index]
		return str(os.path.join(self.data_path, self.set_+"/" + self.set_ + "_images/"+ID+".jpg"))
    """
    object_detector = ObjectDetector()
    # threshold (0.4) is the confidence value above which to consider a box relevant ; boxes and class_label are as expected
    boxes, class_label = object_detector(inference_example, 0.5)
    object_detector.display_objects(inference_example, boxes, class_label, show_label = True)
    save_dir = 'inference_examples'
    os.makedirs(save_dir, exist_ok = True)
    object_detector.save_boxed_image(inference_example, save_dir, boxes, class_label, show_label = False)