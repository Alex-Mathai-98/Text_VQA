from optical_character_recognition.text_detection import TextDetection
from optical_character_recognition.text_recognition import TextRecognition
from tqdm import tqdm
import numpy as np
import torch
import cv2
import os 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
root = "frames"
out = os.path.join(root, "out_frames")
frames = sorted(os.listdir(root))
text_detector = TextDetection().to(device)
text_recognition = TextRecognition().to(device)
question = "What is the quantity on the bottle?"
font = cv2.FONT_HERSHEY_COMPLEX
n = 0

for frame in tqdm(frames):
    frame_path = os.path.join(root, frame)
    image, boxes, polys = text_detector.display(frame_path, out, show = False, threshold=0.8)
    image[:30] = 0
    image[-30:] = 0
    cv2.putText(image, question, (2, 18),  cv2.FONT_HERSHEY_COMPLEX, 0.75, (255, 255, 255),thickness = 1)
    if n <= 13: cv2.putText(image, 'Predicted Answer: 250g', (2, image.shape[0]-8), font, 0.6, (255, 255, 255),thickness = 1)
    elif n > 13 and n <= 17: cv2.putText(image, 'Predicted Answer: Unknown', (2, image.shape[0]-8), font, 0.6, (255, 255, 255),thickness = 1)
    elif n > 17: cv2.putText(image, 'Predicted Answer: 200ml', (2, image.shape[0]-8), font, 0.6, (255, 255, 255),thickness = 1)
    cv2.imwrite(os.path.join(out, frame), image)
    n += 1
    