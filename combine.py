import csv
import os
import torch
import torchvision.models as models
from imageTestDataSet import get_image_only_dataset
from voiceTestDataSet import get_audio_only_dataset
import librosa
import torchvision.transforms as transforms
import torch.nn as nn


# 准备声音数据
def preprocess_voice(audio_file_path):
    audio, sr = librosa.load(audio_file_path, sr=None)
    mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr)
    return mel_spectrogram

transform = transforms.Compose([
    transforms.Resize(250),  # 调整图像大小为 224x224
    transforms.ToTensor(),   # 转换为张量
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
])
pineapple_id = []
for i in range(300,485):
     pineapple_id.append(i)
if  not os.path.isfile("./newAnswer.csv"):
    with open('newAnswer.csv', 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['ID', 'ImageLabel', 'voiceLabel', 'combineLabel']) # 写入标题行

    file.close()

imagedataset = get_image_only_dataset(transform=transform)
voicedataset = get_audio_only_dataset(transform=None)

voiceDataloder = torch.utils.data.DataLoader(voicedataset, batch_size=1)
imageDataloader = torch.utils.data.DataLoader(imagedataset, batch_size=1)

# 加载训练好的声音分类模型和图像分类模型
image_model = torch.load('resnext50ImageVGreyMAX.pth')
voice_model = torch.load('resnext50VoiceMAXV3.pth')


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_model.to(device)
voice_model.to(device)
# 将模型设置为评估模式
image_model.eval()
voice_model.eval()
with torch.no_grad():
    softmax = nn.Softmax(dim=1)
    index = 0
    imgcorrect = 0
    voicecorrect = 0
    correct = 0
    '''for i in range(len(imageDataloader)):
        print(f'{i} / {len(imageDataloader.dataset)}')
        images = imageDataloader.dataset[i]
        voice = voiceDataloder.dataset[i]'''
    for images, voice in zip(imageDataloader,voiceDataloder):
        #label = imageDataloader.dataset.pineapples[index].label
        #label.to(device)
        #_, label = label.max(0)
        sum_tensor = torch.zeros(1,4)
        sum_tensor = sum_tensor.to(device)
        for img in images:
            img = img.to(device)
            outputs = image_model(img)
            outputs = softmax(outputs)
            sum_tensor += outputs
            _, predicted = outputs.max(1)

        _, answer = sum_tensor.max(1)
        imageAnswer = sum_tensor

        #if answer.item() == label.item():
        #    imgcorrect += 1
        sum_tensor = torch.zeros(1,4)
        sum_tensor = sum_tensor.to(device)
        for audio in voice:
            audio = audio.to(device)
            outputs = voice_model(audio)
            outputs = softmax(outputs)
            sum_tensor += outputs
            _, predicted = outputs.max(1)

        _, answer = sum_tensor.max(1)
        audoiAnswer = sum_tensor
        #if answer.item() == label.item():
        #    voicecorrect += 1

        answer = softmax(imageAnswer) * 0.4 + softmax(audoiAnswer) * 0.7
        _, answer = answer.max(1)
        #if answer.item() == label.item():
        #    correct += 1

        #print(answer.item())
        with open('newAnswer.csv', 'a', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                _,imgAns = imageAnswer.max(1)
                _,voiceAns = audoiAnswer.max(1)
                writer.writerow([pineapple_id[index],imgAns.item(),voiceAns.item(),answer.item()])
        file.close()
        index += 1
    #print(f'imgaccuracy: {100. * imgcorrect / len(imageDataloader.dataset)}')
    #print(f'voiceacc: {100. * voicecorrect / len(imageDataloader.dataset)}')
    #print(f'accuracy: {100. * correct / len(imageDataloader.dataset)}')
       # _, answer = sum_tensor.max(1)
