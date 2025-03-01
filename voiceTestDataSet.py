import torch
import os
import librosa
import multiprocessing as mp
import logging
from typing import List
import numpy as np

class PineappleAudioOnlyData:

    def _get_mel_spectrogram(self, audio_file_path):
        audio, sr = librosa.load(audio_file_path, sr=None)
        mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr)
        return mel_spectrogram

    def __init__(self, path, label = None):
        
        self.pineapple_id = str(int(os.path.basename(path)))
        self.voice = []
        # concatenate all the mel spectrograms as a 8 x 128 x 24 tensor
        c = ''
        s = ''
        f = ''
        for cam in range(1,3):
            if (cam == 1):  
                c = 'cam-1/'
            else:
                c = 'cam-2/'
            for buttom in range(0,2):
                if(buttom == 0):
                    s = 'pine-side/'
                else:
                    s = 'pine-bottom/'
                for file in range(1,3):
                    if(file == 1):
                        f = 'cleaned_01.wav'
                        melData = self._get_mel_spectrogram(os.path.join(path, c+s+ 'mic-1/'+f))
                        melData = torch.from_numpy(melData).unsqueeze(0)
                        melData = melData.expand(3, -1, -1)
                        self.voice.append(melData)           
                    elif (file == 2):
                        f = 'cleaned_02.wav'
                        melData = self._get_mel_spectrogram(os.path.join(path, c+s+ 'mic-1/'+f))
                        melData = torch.from_numpy(melData).unsqueeze(0)
                        melData = melData.expand(3, -1, -1)
                        self.voice.append(melData)   
        
        if label != None:
            output_vector = np.zeros(4)
            output_vector[label] = 1
            self.label = torch.from_numpy(output_vector)

class PineappleAudioOnlyDataset(torch.utils.data.Dataset):

    def __init__(self, pineapples: List[PineappleAudioOnlyData], transform=None):
        self.pineapples = pineapples
        self.transform = transform

    def __len__(self):
        return len(self.pineapples)
    
    def __getitem__(self, index):
        if self.transform:
            result = []
            for i in range(len(self.pineapples[index].voice)):
                result.append(self.transform(self.pineapples[index].voice[i]))
            return result

        return self.pineapples[index].voice

def get_audio_only_dataset(transform):
    pineapples = []
    #label_file = 'training/pineapple_training_label.csv'
    label_file = 'training/pineapple_training_label.csv'
    with open(label_file, 'r') as f:
        #for line in f:
        for i in range(300,485):
            # skip the header
            '''if line.startswith('ID'):
                continue
            line = line.strip()
            line = line.split(',')
            pineapple_id = line[0]'''
            pineapple_id = i

            label = None
            #label = int(line[1])
            # pineapple_id to four digit
            pineapple_id = str(int(pineapple_id)).zfill(4)
            pineapple_path = os.path.join('Pineapple_New_Test_Data', pineapple_id)
            pineapple = PineappleAudioOnlyData(pineapple_path, label)
            pineapples.append(pineapple)
            
    dataset = PineappleAudioOnlyDataset(pineapples, transform=transform)
    return dataset

if __name__ == '__main__':
    #logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    dataset = get_audio_only_dataset(transform=None)

    print(dataset[0][0].shape)


        

