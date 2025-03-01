import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim

transform = transforms.Compose([
    transforms.Resize(250),  # 调整图像大小为 224x224
    transforms.ToTensor(),   # 转换为张量
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
])

class history:
    def __init__(self):
        self.history = {}

#定义损失和精度的图,和混淆矩阵指标等等
from matplotlib import pyplot as plt
def plot_loss(history):
    # 显示训练和验证损失图表
    plt.subplots(1,2,figsize=(10,3))
    plt.subplot(121)
    loss = history.history["loss"]
    epochs = range(1, len(loss)+1)
    val_loss = history.history["val_loss"]
    plt.plot(epochs, loss, "bo", label="Training Loss")
    plt.plot(epochs, val_loss, "r", label="Validation Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()  
    plt.subplot(122)
    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    plt.plot(epochs, acc, "b-", label="Training Acc")
    plt.plot(epochs, val_acc, "r--", label="Validation Acc")
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    #plt.show()
    plt.savefig("trainVoice.png")

from dataset import get_audio_only_dataset
#from imagedataset import get_image_only_dataset

dataset = get_audio_only_dataset(transform=None)
#dataset = get_image_only_dataset(transform=transform)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True)

model = torchvision.models.resnext50_32x4d(weights='ResNeXt50_32X4D_Weights.DEFAULT')
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 4)  # 将输出层的数量设置为 4（根据您的分类任务）

#model = torch.load('resnext50VoiceVer2.pth')
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
history = history()
lossHistory = []
valLossHistory = []
accHistory = []
valAccHistory = []
max_accuracy = 69.444
for epoch in range(20):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        #print(images.shape)
        optimizer.zero_grad()
        outputs = model(images)
        #print(outputs.shape)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        softmax = nn.Softmax(dim=1)
        outputs = softmax(outputs)
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        _, answer = labels.max(1)
        total += labels.size(0)
        correct += predicted.eq(answer).sum().item()
    lossHistory.append(running_loss / len(dataloader))
    accHistory.append(100. * correct / total)
    print('[%d] loss: %.3f accuracy: %.3f' % (epoch + 1, running_loss / len(dataloader), 100. * correct / total))

    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        losses = 0.0
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            losses += loss.item()
            softmax = nn.Softmax(dim=1)
            outputs = softmax(outputs)
            _, predicted = outputs.max(1)
            _, answer = labels.max(1)
            total += labels.size(0)
            correct += predicted.eq(answer).sum().item()


        valLossHistory.append(losses / len(testloader))
        valAccHistory.append(100. * correct / total)
        print('test loss: %.3f accuracy: %.3f' % (losses / len(testloader), 100. * correct / total))
        if(max_accuracy < 100. * correct / total):
            max_accuracy = 100. * correct / total
            torch.save(model, 'resnext50VoiceMAXVALL.pth')
            print("save")

torch.save(model, 'resnext50VoiceVerALL.pth')
history.history['loss'] = lossHistory
history.history['val_loss'] = valLossHistory
history.history['accuracy'] = accHistory
history.history['val_accuracy'] = valAccHistory
plot_loss(history)

