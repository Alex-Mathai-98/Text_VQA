## Optical Character Recognition

This folder is to contain all code relevant to the OCR model

## Text Detection

Once text is detected and obtained at letter level, then the OCR problem becomes a classification problem. The following is the text detection model that currently works best.  

To run the model, run the .py file as a module as done for object recognition. Image for inference is chosen at random from the folder of images given 

Run the below line from `Text_VQA` as current dir

    >>> python3 -m optical_character_recognition.text_detection --test_folder=/path/to/folder/of/images

It was published in last years CVPR and the model is called CRAFT. It works extremely well, but there probably is a problem with the way I have changed the code, as on CPU (mine) it takes a while to infer (might be the method itself, idk). I feel its a bug somewhere, will fix soon and extend to full OCR. Some examples are below.

Image with boxes           |  Heat map of letter probs
:-------------------------:|:-------------------------:
![](https://github.com/Alex-Mathai-98/Text_VQA/blob/master/optical_character_recognition/inference_examples/detection_754c36b7179116de.jpg)  |  ![](https://github.com/Alex-Mathai-98/Text_VQA/blob/master/optical_character_recognition/inference_examples/letter_heatmap_754c36b7179116de.jpg)

Image with boxes           |  Heat map of letter probs
:-------------------------:|:-------------------------:
![](https://github.com/Alex-Mathai-98/Text_VQA/blob/master/optical_character_recognition/inference_examples/detection_b00478eb3c722013.jpg)  |  ![](https://github.com/Alex-Mathai-98/Text_VQA/blob/master/optical_character_recognition/inference_examples/letter_heatmap_b00478eb3c722013.jpg)

## Text Recognition 
Once text detection is done, the polygons are cropped containing the text, and passed through a text recognition model. Text recognition model used is CRNN. Run using - 

    >>> python -m optical_character_recognition.end_to_end_ocr # For random image
    >>> python -m optical_character_recognition.end_to_end_ocr --image_path=/path/to/img/or/dir # If img then that image, if dir then random image within that dir
  
Some demonstrations are below
| ![dialogue](https://github.com/Alex-Mathai-98/Text_VQA/blob/master/optical_character_recognition/inference_examples/demonstration2.png) |
|:-------------------------------------------------------------------------------------------------:|
| Demonstration 1 |

| ![dialogue](https://github.com/Alex-Mathai-98/Text_VQA/blob/master/optical_character_recognition/inference_examples/demonstration1.png) |
|:-------------------------------------------------------------------------------------------:|
| Demonstration 2 |
   
Some results of the complete end to end OCR pipeline are below -
