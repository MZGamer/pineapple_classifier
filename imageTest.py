import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn
from imageTestDataSet import get_image_only_dataset
transform = transforms.Compose([
    transforms.Resize(250),  # 调整图像大小为 224x224
    transforms.ToTensor(),   # 转换为张量
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
])

dataset = get_image_only_dataset(transform=transform)


dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)

# 加载训练好的声音分类模型和图像分类模型
image_model = torch.load('resnext50Image.pth')
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
        
