import json
import os

set_ = "test"
from_data_path = "../Data/" + set_ + "/meta-data.json"
to_data_path = "../Data/" + set_ + "/cleaned.json"

def getAnswers(answers):

	max_cnt = -1
	max_answer = ""
	max_cnt_dict = {}

	for key in answers:

		key = key.lower()

		if key == "unanswerable":
			continue

		if max_cnt_dict.get(key,-1) == -1 :
			max_cnt_dict[key] = [key,1]
		else :
			max_cnt_dict[key][1] += 1

		if max_cnt_dict[key][1] > max_cnt :
			max_cnt = max_cnt_dict[key][1]
			max_answer = key

	return max_cnt_dict[max_answer]

with open(from_data_path) as f:
  data = json.load(f)

data = data['data']

new_json = {"image_classes": {},
			"image_width": {},
			"image_height" : {},
			"ocr_tokens" : {},
			"question_tokens": {},
			"question" : {},
			"answers" : {}}

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
	question = value["question"]
	answers = None

	if value.get("answers",-1) != -1 :
		answers = getAnswers(value["answers"])
		if (answers[1]<3) or (answers[0]=="answering does not require reading text in the image") :
			continue

	new_json["image_classes"][image_id] = image_classes
	new_json["image_height"][image_id] = image_height
	new_json["image_width"][image_id] = image_width
	new_json["ocr_tokens"][image_id] = ocr_tokens
	new_json["question_tokens"][image_id] = question_tokens
	new_json["question"][image_id] = question

	if answers is not None :
		new_json["answers"][image_id] = answers

	imageIdlist.append(image_id)

print("Num Answers : {}".format(len(imageIdlist)))

with open(to_data_path,"w") as f:
	json.dump(new_json,f,indent=4)

with open(os.path.join("../Data/" + set_ + "/", set_ + "_ids.txt"),"w") as f:
	for imageId in imageIdlist :
		f.write(imageId+"\n")