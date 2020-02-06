import json
import os

from_data_path = "../Data/test/meta-data.json"
to_data_path = "../Data/test/cleaned.json"

with open(from_data_path) as f:
  data = json.load(f)

data = data['data']

new_json = {"image_classes": {},
			"image_width": {},
			"image_height" : {},
			"ocr_tokens" : {},
			"question_tokens": {}}

imageIdlist = []
for key,value in enumerate(data) :
	print("Key : {}".format(key))
	print("Value : {}".format(value))

	image_id = value["image_id"]
	image_classes = value["image_classes"]
	image_height = value["image_height"]
	image_width = value["image_width"]
	ocr_tokens = value["ocr_tokens"]
	question_tokens = value["question_tokens"]

	new_json["image_classes"][image_id] = image_classes
	new_json["image_height"][image_id] = image_height
	new_json["image_width"][image_id] = image_width
	new_json["ocr_tokens"][image_id] = ocr_tokens
	new_json["question_tokens"][image_id] = question_tokens

	imageIdlist.append(image_id)

with open(to_data_path,"w") as f:
	json.dump(new_json,f,indent=4)

with open(os.path.join("../Data/test/","test_ids.txt"),"w") as f:
	for imageId in imageIdlist :
		f.write(imageId+"\n")