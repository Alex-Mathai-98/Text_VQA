import json

f = json.load(open("meta-data.json", "r"))

data = f['data']

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

count = 0

for key, value in enumerate(data):
	if value.get("answers",-1) != -1 :
	    answers = getAnswers(value["answers"])
	    if (answers[1]<3) or (answers[0]=="answering does not require reading text in the image") :
	       count += 1 
	       continue

print (count)
