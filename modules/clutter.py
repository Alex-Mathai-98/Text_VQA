file = open("modules/labels.txt")
labels_list = []
num_labels = 182
for label in file.read().splitlines():  
    labels_list.append(label.rsplit(": ")[1])
print(labels_list)