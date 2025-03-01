import torch
import os
import librosa
import multiprocessing as mp
import logging
from typing import List
import numpy as np
from PIL import Image

class PineappleImageData:

    def __init__(self, path, label = None):
        
        self.pineapple_id = str(int(os.path.basename(path)))
        # concatenate all the mel spectrograms as a 8 x 128 x 24 tensor
        self.image = []
        left = 100
        top = 100
        right = 900
        bottom = 900
        s = ''
        c = ''
        imageNum = 0
        for cam in range(1,3):
            if (cam == 1):  
                c = 'cam-1/'
                '''left = 350
                top = 50
                right = 650
                bottom = 350'''
                left = 325
                top = 75
                right = 575
                bottom = 325
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
                        self.image.append(Image.open(os.path.join(path, c + s + 'img/01.JPG')).convert('L').crop((left, top, right, bottom)).convert("RGB"))
                                                
                    elif (num == 2):
                        try:
                            self.image.append(Image.open(os.path.join(path, c + s + 'img/03.JPG')).convert('L').crop((left, top, right, bottom)).convert("RGB"))
                        except:
                            self.image.append(Image.open(os.path.join(path, c + s + 'img/01.JPG')).convert('L').crop((left, top, right, bottom)).convert("RGB"))
                    imageNum += 1
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
    
    def __getitem__(self, index):
        if self.transform:
            result = []
            for i in range(len(self.pineapples[index].image)):
                result.append(self.transform(self.pineapples[index].image[i]))
            return result

        return self.pineapples[index].vector
    
def get_image_only_dataset(transform):
    pineapples = []
    #label_file = 'trainingV2/pineapple_training_labelV2.csv'
    label_file = 'training/pineapple_training_label.csv'
    with open(label_file, 'r') as f:
        #for line in f:
        for i in range(300,485):
            # skip the header
            '''if line.startswith('ID'):
                continue
            line = line.strip()
            line = line.split(',')'''
            #pineapple_id = line[0]
            pineapple_id = i

            label = None
            #label = int(line[1])
            # pineapple_id to four digit
            pineapple_id = str(int(pineapple_id)).zfill(4)
            pineapple_path = os.path.join('Pineapple_New_Test_Data', pineapple_id)
            try:
                pineapple = PineappleImageData(pineapple_path,label)
            except Exception as e:
                print(e)
                continue
            pineapples.append(pineapple)

    print(len(pineapples))
    dataset = PineappleImageDataset(pineapples, transform=transform)
    return dataset