## Rough description of object-detection
The below image is an example of the output of fasterrcnn/maskrcnn.   

fasterrcnn as model type would produce information relevant to the image on the left (returns bounding boxes and corresponding class labels and confidence scores for each bounding box) and maskrcnn would additionally create masks within each of these bounding boxes as seen on the right (returns same thing as fasterrcnn with + a non-binary mask for each bounding box)   

![image](https://github.com/Alex-Mathai-98/Text_VQA/blob/master/image_features/inference_examples/5add8b642058ffda.jpg)

`type = fasterrcnn` would return a tuple of N bounding boxes (N, 4), N predictions and `type = maskrcnn` would return an additional element masks (N, *image_size)
