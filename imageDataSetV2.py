import random
import torch
import os
import librosa
import multiprocessing as mp
import logging
from typing import List
import numpy as np
from PIL import Image

class PineappleImageData:

    def __init__(self, path, label, flip = 0):
        
        self.pineapple_id = str(int(os.path.basename(path)))
        # concatenate all the mel spectrograms as a 8 x 128 x 24 tensor
        self.image = []
        left = 100
        top = 100
        right = 900
        bottom = 900
        s = ''
        c = ''
        self.imageNum = 0
        for cam in range(1,3):
            if (cam == 1):  
                c = 'cam-1/'
                left = 350
                top = 50
                right = 650
                bottom = 350
            else:
                left = 100
                top = 100
                right = 900
                bottom = 900
                c = 'cam-2/'
            for buttom in range(0,2):
                if(buttom == 0):
                    s = 'pine-side/'
                else:
                    s = 'pine-bottom/'
                for num in range(1,3):
                    if(num == 1):
                        self.image.append(Image.open(os.path.join(path, c + s + 'img/01.JPG')).convert('L').crop((left, top, right, bottom)))
                        self.image[self.imageNum] =  self.image[self.imageNum].convert("RGB")        
                    elif (num == 2):
                        try:
                            self.image.append(Image.open(os.path.join(path, c + s + 'img/03.JPG')).convert('L').crop((left, top, right, bottom)))
                            self.image[self.imageNum] =  self.image[self.imageNum].convert("RGB")  
                        except:
                            self.image.append(Image.open(os.path.join(path, c + s + 'img/01.JPG')).convert('L').crop((left, top, right, bottom)))
                            self.image[self.imageNum] =  self.image[self.imageNum].convert("RGB")  
                        for flip in range(0,3):
                            if(label == 2 and flip != 0):
                                break
                            if(label == 0 and flip == 2):
                                break
                            if(flip == 1):
                                self.image.append(self.image[self.imageNum].transpose(Image.FLIP_LEFT_RIGHT))
                                self.imageNum += 1
                            elif(flip == 2):
                                self.image.append(self.image[self.imageNum].transpose(Image.FLIP_TOP_BOTTOM))
                                self.imageNum += 1
                    
                    self.imageNum += 1
        
        if label != None:
            output_vector = np.zeros(4)
            output_vector[label] = 1
            self.label = torch.from_numpy(output_vector)



class PineappleImageDataset(torch.utils.data.Dataset):

    def __init__(self, pineapples: List[PineappleImageData], transform=None):
        self.pineapples = pineapples
        self.transform = transform

    def __len__(self):
        return len(self.pineapples)

    def sizeOfImg(self):
        result = 0
        for pineapple in self.pineapples:
            result += pineapple.imageNum
        return result
    
    def __getitem__(self, index):
        if self.transform:
            result = []
            for i in range(len(self.pineapples[index].image)):
                result.append([self.transform(self.pineapples[index].image[i]), self.pineapples[index].label])
            return result
        result = []
        for i in range(len(self.pineapples[index].image)):
            result.append([self.pineapples[index].image[i], self.pineapples[index].label])
        return result
    
def get_image_only_dataset(transform):
    pineapples = []
    label_file = 'trainingV2/pineapple_training_labelV2.csv'
    with open(label_file, 'r') as f:
        labelCount = [0,0,0,0]
        for line in f:
            # skip the header
            if line.startswith('ID'):
                continue
            line = line.strip()
            line = line.split(',')
            pineapple_id = line[0]

            label = int(line[1])
            # pineapple_id to four digit
            pineapple_id = str(int(pineapple_id)).zfill(4)
            pineapple_path = os.path.join('trainingV2', pineapple_id)
            try:
                pineapple = PineappleImageData(pineapple_path,label)
                labelCount[label] += pineapple.imageNum
            except Exception as e:
                print(e)
                continue
            pineapples.append(pineapple)

    print(len(pineapples))
    dataset = PineappleImageDataset(pineapples, transform=transform)
    trainDataSet = []
    testDataSet = []
    imageNum = dataset.sizeOfImg()
    print(imageNum)
    print(labelCount)
    print(sum(labelCount))
    trainLabelAssumeCount = [labelCount[0] * 0.8, labelCount[1] * 0.8, labelCount[2] * 0.8, labelCount[3] * 0.8]
    trainLabelCount = [0,0,0,0]

    testLabelAssumeCount = [labelCount[0] * 0.2, labelCount[1] * 0.2, labelCount[2] * 0.2, labelCount[3] * 0.2]
    testLabelCount = [0,0,0,0]
    for pineapple in dataset:
        _, current = pineapple[0][1].max(0)
        testOrTrain = np.random.randint(0, 2)
        if(testOrTrain == 0):
            if(trainLabelCount[current] < trainLabelAssumeCount[current]):
                for i in range(len(pineapple)):
                    trainDataSet.append([pineapple[i][0], pineapple[i][1]])
                    trainLabelCount[current] += 1
            else:
                for i in range(len(pineapple)):
                    testDataSet.append([pineapple[i][0], pineapple[i][1]])
                    testLabelCount[current] += 1
        else:
            if(testLabelCount[current] < testLabelAssumeCount[current]):
                for i in range(len(pineapple)):
                    testDataSet.append([pineapple[i][0], pineapple[i][1]])
                    testLabelCount[current] += 1
            else:
                for i in range(len(pineapple)):
                    trainDataSet.append([pineapple[i][0], pineapple[i][1]])
                    trainLabelCount[current] += 1
    return trainDataSet,testDataSet