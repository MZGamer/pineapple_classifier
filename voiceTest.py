import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn
from voiceTestDataSet import get_audio_only_dataset

dataset = get_audio_only_dataset(transform=None)


dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)

# 加载训练好的声音分类模型和图像分类模型
image_model = torch.load('resnext50Voice.pth')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_model.to(device)
# 将模型设置为评估模式
image_model.eval()
with torch.no_grad():
    ind = 0
    for images in dataloader:
        sum_tensor = torch.zeros(1,4)
        sum_tensor = sum_tensor.to(device)
        for img in images:
            img = img.to(device)
            outputs = image_model(img)
            softmax = nn.Softmax(dim=1)
            outputs = softmax(outputs)
            sum_tensor += outputs
            _, predicted = outputs.max(1)

        _, answer = sum_tensor.max(1)
        ind+= 1
        print(answer) 
        
