import os
import csv

test_label_file = 'test/pineapple_old_test_label.csv'
train_label_file = 'training/pineapple_training_label.csv'
pineapple_training_labelV2 = 'trainingV2/pineapple_training_labelV2.csv'


if  not os.path.isfile(pineapple_training_labelV2):
    with open(pineapple_training_labelV2, 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['ID','Label']) # 写入标题行

    file.close()

with open(test_label_file, 'r') as f:
    for line in f:
        # skip the header
        if line.startswith('ID'):
            continue
        line = line.strip()
        line = line.split(',')
        with open(pineapple_training_labelV2, 'a', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow([line[0],line[1]])
        file.close()

with open(train_label_file, 'r') as f:
    for line in f:
        # skip the header
        if line.startswith('ID'):
            continue
        line = line.strip()
        line = line.split(',')
        with open(pineapple_training_labelV2, 'a', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow([line[0],line[1]])
        file.close()