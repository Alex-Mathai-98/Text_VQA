from PIL import Image
import os
from torchvision import transforms
import torch

folder_path = "../Data/test/test_images"
folder_save = "../Data/test/clean_images"

image_list = os.listdir(folder_path)
image_list = [image for image in image_list if ".pt" not in image]
image_list.sort()

transformer = transforms.ToTensor()

for image in image_list :

	image_path = os.path.join(folder_path,image)
	print("Image Path : {}".format(image_path))

	save_path = os.path.join(folder_save,image)[:-4] + ".pt"
	print("Save Path : {}".format(save_path))

	arr = Image.open(image_path)

	torch_arr = transformer(arr)

	print("Array Dimension : {}".format(torch_arr.shape))
	print("Max : {}".format(torch.max(torch_arr)))
	print("Min : {}\n".format(torch.min(torch_arr)))

	torch.save(torch_arr,save_path)