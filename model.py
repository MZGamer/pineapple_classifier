import torch
import torch.nn as nn

class PineappleAudioOnlyModel(nn.Module):
    def __init__(self):
        super(PineappleAudioOnlyModel, self).__init__()
        # (batch_size, 8, 128, 24)

        self.conv = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=(1, 1)), # (batch_size, 32, 128, 24)
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.MaxPool2d(2), # (batch_size, 32, 64, 12)
            nn.Conv2d(32, 128, 3, padding=(1, 1)), # (batch_size, 128, 64, 12)
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.MaxPool2d(2), # (batch_size, 128, 32, 6)
            nn.Conv2d(128, 256, 3, padding=(1, 1)), # (batch_size, 256, 32, 6)
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        self.fc = nn.Linear(256 * 32 * 6, 4)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 256 * 32 * 6)
        x = self.fc(x)
        return x
    

if __name__ == '__main__':

    from torchvision import transforms
    from dataset import get_audio_only_dataset
    BATCH_SIZE = 1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PineappleAudioOnlyModel()
    model.to(device)
    dataset = get_audio_only_dataset(transform=None)

    # train
    from torch.utils.data import DataLoader
    from torch.utils.data import random_split
    from torch.optim import Adam
    from torch.nn import CrossEntropyLoss
    from tqdm import tqdm
    import numpy as np
    

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    optimizer = Adam(model.parameters(), lr=0.001)
    criterion = CrossEntropyLoss()

    def accuracy(y_pred, y_true):
        y_pred = torch.argmax(y_pred, dim=1)
        y_true = torch.argmax(y_true, dim=1)
        return torch.sum(y_pred == y_true).float() / len(y_true)

    for epoch in range(5):
        model.train()
        loss_history = []
        acc_history = []

        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            loss_history.append(float(loss))
            acc = accuracy(outputs, labels)
            acc_history.append(acc)

        print(f'[Train]Epoch {epoch + 1} | Loss: {torch.mean(torch.Tensor(loss_history)):.4f} | Acc: {torch.mean(torch.Tensor(acc_history)):.4f}')

        model.eval()
        loss_history = []
        acc_history = []

        with torch.no_grad():
            for i, data in enumerate(test_loader, 0):
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                loss_history.append(float(loss))
                acc = accuracy(outputs, labels)
                acc_history.append(acc)

        print(f'[Test]Epoch {epoch + 1} | Loss: {torch.mean(torch.Tensor(loss_history)):.4f} | Acc: {torch.mean(torch.Tensor(acc_history)):.4f}')

