import torch
import os
import librosa
import multiprocessing as mp
import logging
from typing import List
import numpy as np
from PIL import Image

class PineappleImageData:

    def __init__(self, path, label, num, cam, button, flip = 0):
        
        self.pineapple_id = str(int(os.path.basename(path)))
        # concatenate all the mel spectrograms as a 8 x 128 x 24 tensor
        left = 100
        top = 100
        right = 900
        bottom = 900
        s = ''
        if (cam == 1):  
            s = 'cam-1/'
            left = 350
            top = 50
            right = 650
            bottom = 350
        else:
            s = 'cam-2/'
        if(button == 0):
            s += 'pine-side/'
        else:
            s += 'pine-bottom/'
        if(num == 1):
            self.vector = Image.open(os.path.join(path, s + 'img/01.JPG')).crop((left, top, right, bottom))
        elif (num == 2):
            self.vector = Image.open(os.path.join(path, s + 'img/03.JPG')).crop((left, top, right, bottom))
        elif (num == 3):
            self.vector = Image.open(os.path.join(path, s + 'img/03.JPG')).crop((left, top, right, bottom))
        elif (num == 4):
            self.vector = Image.open(os.path.join(path, s + 'img/04.JPG')).crop((left, top, right, bottom))
        if(flip == 1):
            self.vector.transpose(Image.FLIP_LEFT_RIGHT)
        elif(flip == 2):
            self.vector.transpose(Image.FLIP_TOP_BOTTOM)
        output_vector = np.zeros(4)
        output_vector[label] = 1
        self.label = torch.from_numpy(output_vector)

class PineappleImageDataset(torch.utils.data.Dataset):

    def __init__(self, pineapples: List[PineappleImageData], transform=None):
        self.pineapples = pineapples
        self.transform = transform

    def __len__(self):
        return len(self.pineapples)
    
    def __getitem__(self, index):
        if self.transform:
            return self.transform(self.pineapples[index].vector), self.pineapples[index].label

        return self.pineapples[index].vector, self.pineapples[index].label
    
def get_image_only_dataset(transform):
    pineapples = []
    label_file = 'test/pineapple_old_test_label.csv'
    
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
            pineapple_path = os.path.join('test', pineapple_id)
            #labelCount[label] += 1
            for num in range(1,3):
                for cam in range(1,3):
                    for button in range(0,2):
                        for flip in range(0,3):
                            if(label == 2 and flip != 0):
                                break
                            if(label == 0 and flip == 2):
                                break
                            try:
                                pineapple = PineappleImageData(pineapple_path, label, num, cam, button, flip)
                                labelCount[label] += 1
                            except Exception as e:
                                print(e)
                                continue
                            pineapples.append(pineapple)

    print(len(pineapples))
    print(labelCount)
    print(sum(labelCount))
    dataset = PineappleImageDataset(pineapples, transform=transform)
    return dataset