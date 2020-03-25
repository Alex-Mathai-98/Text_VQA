import torch, torch.nn as nn
import os, random, argparse, cv2, numpy as np
from optical_character_recognition.text_detection import TextDetection
from optical_character_recognition.text_recognition import TextRecognition

class EndToEndOCR(nn.Module):
    def __init__(self):
        super().__init__()
        self.text_detector = TextDetection()
        self.text_recognizer = TextRecognition()
        if torch.cuda.is_available():
            self.text_detector.cuda(), self.text_recognizer.cuda()
    
    def forward(self, image_path):
        ocr_tokens = list()
        image = cv2.imread(image_path)
        boxes, _, _ = self.text_detector(image_path)
        boxes = boxes.astype(np.int32)
        for box in boxes:
            x_max, x_min = max(box[:, 0]), min(box[:, 0])
            y_max, y_min = max(box[:, 1]), min(box[:, 1])
            tmp = image[y_min:y_max, x_min:x_max]
            _, sim_pred = self.text_recognizer(tmp)
            ocr_tokens.append(sim_pred)
        return ocr_tokens
    
    def visualize_predictions(self, image_path):
        image = cv2.imread(image_path)
        boxes, _, _ = self.text_detector(image_path)
        boxes = boxes.astype(np.int32)
        for box in boxes:
            x_max, x_min = max(box[:, 0]), min(box[:, 0])
            y_max, y_min = max(box[:, 1]), min(box[:, 1])
            cv2.polylines(image, [box], True, (0, 0, 255))
            tmp = image[y_min:y_max, x_min:x_max]
            raw_pred, sim_pred = self.text_recognizer(tmp)
            print("Predicted raw word ======> {} ======> {}".format(raw_pred, sim_pred))
            cv2.imshow("Token", tmp)
            cv2.waitKey()
        cv2.imshow("Complete Image", image)
        cv2.waitKey()

if __name__ == '__main__':

    # Parse single argument for custom image path
    parser = argparse.ArgumentParser(description='Complete end-to-end OCR inference')
    parser.add_argument('--image_path', default='Data/test/test_images/', type=str, help='folder path to input images')
    args = parser.parse_args()

    # If directory, get random image in that directory otherwise just use the image path given
    if os.path.isdir(args.image_path):
        image_path = os.path.join(args.image_path, random.choice(os.listdir(args.image_path)))
    else: 
        image_path = args.image_path

    # Instantiate OCR model, get tokens in the image (acc. to the model), visualize what the model saw as what
    ocr = EndToEndOCR()
    tokens_in_image = ocr(image_path)
    ocr.visualize_predictions(image_path)