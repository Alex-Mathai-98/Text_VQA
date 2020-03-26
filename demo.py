import sys
sys.path.append('fastText/python')
sys.path.append('pythia')
sys.path.append('vqa-maskrcnn-benchmark')

import yaml
import cv2
import torch
import requests
import numpy as np
import gc
import torch.nn.functional as F
import pandas as pd
import base64
import subprocess

import torchvision.models as models
import torchvision.transforms as transforms

from PIL import Image
from IPython.display import display, HTML, clear_output
from ipywidgets import widgets, Layout
from io import BytesIO

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.layers import nms
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.utils.model_serialization import load_state_dict
from optical_character_recognition.end_to_end_ocr import EndToEndOCR

from pythia.utils.configuration import ConfigNode
from pythia.tasks.processors import Processor
from pythia.models.lorra import LoRRA
from pythia.common.registry import registry
from pythia.common.sample import Sample, SampleList

class Writer:
  def write(self, text, level="info"):
    print(text)

class LoRRADemo:
  TARGET_IMAGE_SIZE = [448, 448]
  CHANNEL_MEAN = [0.485, 0.456, 0.406]
  CHANNEL_STD = [0.229, 0.224, 0.225]
  
  def __init__(self):
    gc.collect()
    registry.register("writer", Writer())
  
  def build(self):
    self._init_processors()
    self.pythia_model = self._build_pythia_model().cuda()
    self.detection_model = self._build_detection_model().cuda()
    self.resnet_model = self._build_resnet_model().cuda()
       
  def _init_processors(self):
    config = self.config
    vqa_config = config.task_attributes.vqa.dataset_attributes.textvqa
    text_processor_config = vqa_config.processors.text_processor
    answer_processor_config = vqa_config.processors.answer_processor
    
    text_processor_config.params.vocab.vocab_file = "demo_model_data/vocabulary_100k.txt"
    answer_processor_config.params.vocab_file = "demo_model_data/answers_lorra.txt"
    
    # Add preprocessor as that will needed when we are getting questions from user
    self.text_processor = Processor(text_processor_config)
    self.answer_processor = Processor(answer_processor_config)
    
    registry.register("textvqa_text_processor", self.text_processor)
    registry.register("textvqa_answer_processor", self.answer_processor)
    registry.register("textvqa_num_final_outputs", 
                      self.answer_processor.get_vocab_size())
  
  def init_ocr_processor(self):
    with open("demo_model_data/lorra.yaml") as f:
      config = yaml.load(f)
    
    config = ConfigNode(config)
    # Remove warning
    config.training_parameters.evalai_inference = True
    registry.register("config", config)
    
    vqa_config = config.task_attributes.vqa.dataset_attributes.textvqa
    
    self.config = config
    ocr_token_processor_config = vqa_config.processors.ocr_token_processor
    self.ocr_token_processor = EndToEndOCR()
    
  def _build_pythia_model(self):
    state_dict = torch.load('demo_model_data/lorra.pth')
    model_config = self.config.model_attributes.lorra
    model_config.model_data_dir = "."
    model = LoRRA(model_config)
    model.build()
    model.init_losses_and_metrics()
    
    if list(state_dict.keys())[0].startswith('module') and \
       not hasattr(model, 'module'):
      state_dict = self._multi_gpu_state_to_single(state_dict)
          
    model.load_state_dict(state_dict)
    model.to("cuda")
    model.eval()
    
    return model
  
  def _build_resnet_model(self):
    self.data_transforms = transforms.Compose([
        transforms.Resize(self.TARGET_IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(self.CHANNEL_MEAN, self.CHANNEL_STD),
    ])
    resnet152 = models.resnet152(pretrained=True)
    resnet152.eval()
    modules = list(resnet152.children())[:-2]
    self.resnet152_model = torch.nn.Sequential(*modules)
    self.resnet152_model.to("cuda")
  
  def _multi_gpu_state_to_single(self, state_dict):
    new_sd = {}
    for k, v in state_dict.items():
        if not k.startswith('module.'):
            raise TypeError("Not a multiple GPU state of dict")
        k1 = k[7:]
        new_sd[k1] = v
    return new_sd
  
  def predict(self, url, question):
    with torch.no_grad():
      detectron_features = self.get_detectron_features(url)
      resnet_features = self.get_resnet_features(url)

      sample = Sample()
  
      context = self.get_fasttext_vectors()
#       context = self.context_processor({"tokens": ocr_tokens})
      sample.context = context["text"]
      sample.context_tokens = context["tokens"]
      sample.context_feature_0 = context["text"]
      sample.context_info_0 = Sample()
      sample.context_info_0.max_features = torch.tensor(context["length"], 
                                                        dtype=torch.long)
      order_vectors = torch.eye(len(sample.context_tokens))
      order_vectors[context["length"] :] = 0
      sample.order_vectors = order_vectors
      
      processed_text = self.text_processor({"text": question})
      sample.text = processed_text["text"]
      sample.text_len = len(processed_text["tokens"])

      sample.image_feature_0 = detectron_features
      sample.image_info_0 = Sample({
          "max_features": torch.tensor(100, dtype=torch.long)
      })

      sample.image_feature_1 = resnet_features

      sample_list = SampleList([sample])
      sample_list = sample_list.to("cuda")

      scores = self.pythia_model(sample_list)["scores"]
      scores = torch.nn.functional.softmax(scores, dim=1)
      actual, indices = scores.topk(5, dim=1)

      top_indices = indices[0]
      top_scores = actual[0]

      probs = []
      answers = []
      
      answer_space_size = self.answer_processor.get_true_vocab_size()
      
      for idx, score in enumerate(top_scores):
        probs.append(score.item())
        
        answer_id = top_indices[idx].item()
        
        if answer_id >= answer_space_size:
          answer_id -= answer_space_size
          answer = sample.context_tokens[answer_id]
        else:
          answer = self.answer_processor.idx2word(answer_id)
        if answer == "<pad>":
          answer = "unanswerable"
          
        answers.append(answer)
    
    torch.cuda.empty_cache()
    
    return probs, answers
  
  def _get_ocr_tokens(self, url):
      tokens = self.ocr_token_processor(url)
      return tokens

  def dump_fasttext_vectors(self, url):
    ocr_tokens = self._get_ocr_tokens(url)
    ocr_tokens = [self.ocr_token_processor({"text": token})["text"] for token in ocr_tokens]

    print("OCR tokens returned by the ocr module", ocr_tokens)

    with open("tokens.txt", "w") as f:
      f.write("\n".join(ocr_tokens))
      
    cmd = "fastText/fasttext print-word-vectors pythia/pythia/.vector_cache/wiki.en.bin < tokens.txt"
    
    output = subprocess.check_output(cmd, shell=True)
    
    with open("vectors.txt", "w") as f:
      f.write(output.decode("utf-8"))
    
    self.ocr_tokens = ocr_tokens
    
  def get_fasttext_vectors(self):
    with open("vectors.txt", "r") as f:
      output = f.read()
   
    vectors = output.split("\n")
    vectors = [np.array(list(map(float, vector.split(" ")[1:-1])))
               for vector in vectors]
    output = torch.full(
        (50, 300),
        fill_value=0,
        dtype=torch.float,
    )
    
    ocr_tokens = self.ocr_tokens
    length = min(50, len(ocr_tokens))
    ocr_tokens = ocr_tokens[:length]
    
    for idx, token in enumerate(ocr_tokens):
      output[idx] = torch.from_numpy(vectors[idx])
    
    final_tokens = ["<pad>"] * 50
    final_tokens[:length] = ocr_tokens
    ret = {
        "text": output,
        "tokens": final_tokens,
        "length": length
    }
    
    return ret
  
  def _build_detection_model(self):
    cfg.merge_from_file('demo_model_data/detectron_model.yaml')
    cfg.freeze()

    model = build_detection_model(cfg)
    checkpoint = torch.load('demo_model_data/detectron_model.pth', 
                            map_location=torch.device("cpu"))

    load_state_dict(model, checkpoint.pop("model"))

    model.to("cuda")
    model.eval()
    return model
  
  def get_actual_image(self, image_path):
    if image_path.startswith('http'):
      path = requests.get(image_path, stream=True).raw
    else:
      path = image_path
    return path
  
  def _get_image_bytes(self, image_path):
    image_path = self.get_actual_image(image_path)
    
    with BytesIO() as output:
        with Image.open(image_path) as img:
            img.save(output, 'JPEG')
        data = base64.b64encode(output.getvalue()).decode('utf-8')
    return data
    
  def _image_transform(self, image_path):
      path = self.get_actual_image(image_path)

      img = Image.open(path)
      im = np.array(img).astype(np.float32)
      im = im[:, :, ::-1]
      im -= np.array([102.9801, 115.9465, 122.7717])
      im_shape = im.shape
      im_size_min = np.min(im_shape[0:2])
      im_size_max = np.max(im_shape[0:2])
      im_scale = float(800) / float(im_size_min)
      # Prevent the biggest axis from being more than max_size
      if np.round(im_scale * im_size_max) > 1333:
           im_scale = float(1333) / float(im_size_max)
      im = cv2.resize(
           im,
           None,
           None,
           fx=im_scale,
           fy=im_scale,
           interpolation=cv2.INTER_LINEAR
       )
      img = torch.from_numpy(im).permute(2, 0, 1)
      return img, im_scale


  def _process_feature_extraction(self, output,
                                 im_scales,
                                 feat_name='fc6',
                                 conf_thresh=0.2):
      batch_size = len(output[0]["proposals"])
      n_boxes_per_image = [len(_) for _ in output[0]["proposals"]]
      score_list = output[0]["scores"].split(n_boxes_per_image)
      score_list = [torch.nn.functional.softmax(x, -1) for x in score_list]
      feats = output[0][feat_name].split(n_boxes_per_image)
      cur_device = score_list[0].device

      feat_list = []

      for i in range(batch_size):
          dets = output[0]["proposals"][i].bbox / im_scales[i]
          scores = score_list[i]

          max_conf = torch.zeros((scores.shape[0])).to(cur_device)

          for cls_ind in range(1, scores.shape[1]):
              cls_scores = scores[:, cls_ind]
              keep = nms(dets, cls_scores, 0.5)
              max_conf[keep] = torch.where(cls_scores[keep] > max_conf[keep],
                                           cls_scores[keep],
                                           max_conf[keep])

          keep_boxes = torch.argsort(max_conf, descending=True)[:100]
          feat_list.append(feats[i][keep_boxes])
      return feat_list

  def masked_unk_softmax(self, x, dim, mask_idx):
      x1 = F.softmax(x, dim=dim)
      x1[:, mask_idx] = 0
      x1_sum = torch.sum(x1, dim=1, keepdim=True)
      y = x1 / x1_sum
      return y
   
  def get_resnet_features(self, image_path):
      path = self.get_actual_image(image_path)
      img = Image.open(path).convert("RGB")
      img_transform = self.data_transforms(img)
      
      if img_transform.shape[0] == 1:
        img_transform = img_transform.expand(3, -1, -1)
      img_transform = img_transform.unsqueeze(0).to("cuda")
      
      features = self.resnet152_model(img_transform).permute(0, 2, 3, 1)
      features = features.view(196, 2048)
      return features
    
  def get_detectron_features(self, image_path):
      im, im_scale = self._image_transform(image_path)
      img_tensor, im_scales = [im], [im_scale]
      current_img_list = to_image_list(img_tensor, size_divisible=32)
      current_img_list = current_img_list.to('cuda')
      with torch.no_grad():
          output = self.detection_model(current_img_list)
      feat_list = self._process_feature_extraction(output, im_scales, 
                                                  'fc6', 0.2)
      return feat_list[0]
      
def execute(image_url, question):
  clear_output()
  demo = LoRRADemo()
  image_path = demo.get_actual_image(image_url)
  image = Image.open(image_path)
  demo.init_ocr_processor()
  demo.dump_fasttext_vectors(image_url)
  print("--------------------Building--------------------")
  demo.build()
  print("--------------------Predicting--------------------")
  scores, predictions = demo.predict(image_url, question)
  
  scores = [score * 100 for score in scores]
  del demo
  
  df = pd.DataFrame({
      "Prediction": predictions,
      "Confidence": scores
  })
  
  display(image)
  print("Question:", question)
  display(HTML(df.to_html()))
  
  gc.collect()
  torch.cuda.empty_cache()

if __name__ == '__main__':

  image_url = 'Data/demos/1b32c573daac7982.jpg' 
  question = "what is written above the door" 

  execute(image_url, question)
