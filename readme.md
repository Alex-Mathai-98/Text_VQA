# Text Visual Question Answering in Natural Scene Imagery 

The following is a description of our model and each module involved in the pipeline with links linking to a better thorough description of each module. In this readme we have only shown some visualizations to be able to interpret what the model is doing and to understand the pipeline better. 

|![](https://github.com/Alex-Mathai-98/Text_VQA/blob/master/diagrams/TextVQA%20(2).jpg) |
|:----:|
| The architecture of our implementation of Text Visual Question Answering | 

|<img src="https://github.com/Alex-Mathai-98/Text_VQA/blob/master/frames/test_out.gif" alt="drawing" width="300"/> |
|:--:|
|A demonstration of the question answering model on a video; to maintain visual quality only the text detection results have been boxed (Bright green boxes); See [here](https://github.com/Alex-Mathai-98/Text_VQA/blob/master/frames/test_out.mp4) for a video version|  

As visible from the repository structure, each module has a folder of its own. Even though all models are described here well, instructions to run each submodule by itself has been mentioned in the readme present in the folder corresponding to that submodule. Hence, this readme (main readme) will not contain code running instructions, since that will be mentioned in the readme of each submodule. 

---

## Text Feature Extraction 
The model needs a representation of the question to understand what is being asked in the first place, and what regions of the image and what OCR tokens need to be paid attention to. For this, we have used transformer based BERT embeddings to get the words and used transformer based self attention to get the representation of the question. 

This question representation is an improvement over the original [papers](https://arxiv.org/abs/1904.08920) LSTM based self-attention on GloVe embeddings. 

This representation helps the model understand the question and is used to focus on particular regions of the image and on particular OCR tokens, that are most relevant to the question. 

Code in `Text_Features`

---

## Image Feature Extraction 
The image feature extraction has two implementations; both of which are used in the final model in conjunction. These two implementations are grid based features and the object level features. 

Code in `image_features`

## Grid based features 
The grid based features are a 14 x 14 grid that are the output of a pre-trained models convolution.  

We have used [ResNet152](https://arxiv.org/abs/1512.03385)'s 14 x 14 x 2048 layer, and average along the channels to be able to get our grid. Each image corresponds exactly one 14 x 14 grid based feature 

## Object Level Features 
For object level features, first the image is passes through an object detection algorithm which gives us k number of objects in that image. Each of these k objects are then cropped and passed through a feature extractor giving us a tensor of shape (k, 2048), where k is different for each image. 

|<img src="https://github.com/Alex-Mathai-98/Text_VQA/blob/master/diagrams/fasterrcnn.png" alt="drawing" width="420"/> |<img src="https://github.com/Alex-Mathai-98/Text_VQA/blob/master/diagrams/maskrcnn.jpg" alt="drawing" width="540"/> |
|:---:|:----:|
|Faster-RCNN sample result| Mask-RCNN sample result |

The pre-trained models we have used are both [faster-rcnn](https://arxiv.org/abs/1506.01497) and [mask-rcnn](https://arxiv.org/abs/1703.06870), based on a flag. The feature extractor we have used is [ResNet152](https://arxiv.org/abs/1512.03385)'s 2048 dimensional layer. 

---

## Optical Character Recognition

The optical character recognition module primarily consists of two stages - the text detection and the text recognition stage. The text detection stage is what tells the OCR module where readable content is; this is followed by post-processing to increase the readibility of the readable content followed by the reading itself (Text Recognition) 

Code in `optical_character_recognition`

## Text Detection 

We use the current state of the art text detection [model](https://arxiv.org/pdf/1904.01941) and further improve it with our post-processing part of the pipeline. 
The output of the model are two heatmaps - the character probability heatmap (region score) and gap probability heatmap (affinity score). 

The character probability heatmap is a feature map, of the same size of the image, giving the probability that each pixel is the center of a letter. A gaussian prior has been imposed on each character for this distribution. The gap probability map is the feature map having probabilities that a particular pixel is the center of the gap between two letters, with a gaussian prior for each gap. 


|<img src="https://github.com/Alex-Mathai-98/Text_VQA/blob/master/Data/demos/tabasco.png" alt="drawing" width="300"/>|<img src="https://github.com/Alex-Mathai-98/Text_VQA/blob/master/optical_character_recognition/inference_examples/%20_screenshot_29.06sds.2020.png" alt="drawing" width="320"/> |<img src="https://github.com/Alex-Mathai-98/Text_VQA/blob/master/optical_character_recognition/inference_examples/%20_screenshot_29.06.2020.png" alt="drawing" width="320"/> |
|:---:|:----:|:----:|
|Original Image|Character Probability heatmap| Gap Probability heatmap |  

## Post-Processing 
Once we have this heatmaps we process the both the feature maps to get text level bounding boxes. These bounding boxes are lenient with gaps (refine = True), meaning our main intent is capture phrases initially. Once we have these phrases we further break these boxes into words and letters. This gives us a three stage heirarchy ; phrases -> words -> letters ; and helps the OCR model read words that are read top to bottom but have letters read from left to right.  

|![](https://github.com/Alex-Mathai-98/Text_VQA/blob/master/optical_character_recognition/inference_examples/_screenshot_01.07.2020.png)|![](https://github.com/Alex-Mathai-98/Text_VQA/blob/master/optical_character_recognition/inference_examples/detection_plane.jpg) |
|:----:|:----:|
|Letter level bounding boxes | Word + Phrase level bounding boxes |

We use this heirarchy to understand the orientation of the image and process the word in the images to be left to right readable  

|<img src="https://github.com/Alex-Mathai-98/Text_VQA/blob/master/optical_character_recognition/inference_examples/_screenshot_01.07.2020.png" alt="drawing" width="490"/> |<img src="https://github.com/Alex-Mathai-98/Text_VQA/blob/master/optical_character_recognition/inference_examples/goodluk.png" alt="drawing" width="490"/> |
|:----:|:----:|
|<img src="https://github.com/Alex-Mathai-98/Text_VQA/blob/master/optical_character_recognition/inference_examples/Crop_screenshot_02.07.2020.png" alt="drawing" width="490"/> |<img src="https://github.com/Alex-Mathai-98/Text_VQA/blob/master/optical_character_recognition/inference_examples/oneway.png" alt="drawing" width="490"/> |
|Normalize rotation for "jet2.com" | Normalize rotation for "one way" |

## Text Recognition
The text recognition module looks at the lowest levels of the heirarchy (either letter level boxes or word level boxes depending on orientation) and reads either from top to bottom or left to right based on the orientation (Assumed that nothing in english is read right-to-left or bottom-to-top). Once this reading is done the text predictions at lowest level are aggregated based on orientation to make the prediction for the higher level and eventually the top level. 

The collections of all the predictions at all levels (apart from letter level) are all put together as the list of ocr tokens for that image. The model used for left to right reading can be found [here](https://arxiv.org/abs/1507.05717).

---

## Multimodal attention 
We now need to fuse all these feature representations to get one representation for the understanding of the question, image, tokens pairs. We have mostly used linear layers and attentions over these representations to evaluate these attended representations. More details can be found in the report

Code in `Multimodal`

---

## Copy Mechanism 
We have implemented our own variation of a copy mechanism that is able to predict answers not only in the answer space but also in the OCR tokens to be able to answer a question from content that was read. More details can be found in the code and the report.

Code in `main.py`

---

## Interactive demo of the Baseline 

As previously demonstrated, we had also built an interactive demo notebook using modules used by the paper and our older OCR pipeline. That can be found [here](https://colab.research.google.com/drive/1iGUMCTaeE79FJaFuxhJHgZXZclPRCnxV?usp=sharing). 

Instructions for running on any image link are presented in the notebook and the notebook and main code are both well documented

This notebook might face some issues due to the changed paths of the source code and downloads in the notebook takes approx ~10 complete minutes 

---

## Concluding Comments

This readme has described each of the induvidual submodules that were built for text visual question answering. 

Each individual submodule can be run by itself and instructions to do so are in the respective folder readme's (If possible to run individually).

Details regarding the vanilla VQA model built and other detials are better described in the paper/report.
