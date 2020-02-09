import json
import os
import shutil

dev_ids = {}
with open(os.path.join("../Data/dev/dev_ids.txt"),"r") as f:
	for line in f :
		dev_ids[line[:-1]] = 1

train_ids = {}
with open(os.path.join("../Data/train/train_ids.txt"),"r") as f:
	for line in f :
		train_ids[line[:-1]] = 1


for key in dev_ids.keys() :

	# source path
	src_path = os.path.join("../Data/train/train_images/",key +".jpg")
	print("Src Path : {}".format(src_path))

	# destination path
	dst_path = os.path.join("../Data/dev/dev_images/",key +".jpg")
	print("Dst Path : {}".format(dst_path))

	# if image only in "dev" then move to "dev"
	if (train_ids.get(key,-1) == -1) and (not os.path.exists(dst_path)) :
		shutil.move(src_path,dst_path)
	# if image in "train" and "dev" then make a copy
	elif (train_ids.get(key,-1) != -1) : 
		shutil.copyfile(src_path, dst_path)