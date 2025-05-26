import torch
import torch.nn as nn


class IrCNN(nn.Module):
    def __init__(self, signal_size=1024, kernel_size=11, in_ch=1, num_classes=17, p=0.48599073736368):
        super(IrCNN, self).__init__()
        self.num_classes = num_classes
        # 1st CNN layer.
        self.CNN1 = nn.Sequential(
            nn.Conv1d(in_channels=in_ch, out_channels=31, kernel_size=kernel_size, stride=1, padding=0),
            nn.BatchNorm1d(num_features=31),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.cnn1_size = int(((signal_size - kernel_size + 1 - 2) / 2) + 1)
        # 2nd CNN layer.
        self.CNN2 = nn.Sequential(
            nn.Conv1d(in_channels=31, out_channels=62, kernel_size=kernel_size, stride=1, padding=0),
            nn.BatchNorm1d(num_features=62),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.cnn2_size = int(((self.cnn1_size - kernel_size + 1 - 2) / 2) + 1)
        # 1st dense layer.
        self.DENSE1 = nn.Sequential(
            nn.Linear(in_features=24738, out_features=4927),
            nn.ReLU(),
            nn.Dropout(p=p)
        )
        # 2st dense layer.
        self.DENSE2 = nn.Sequential(
            nn.Linear(in_features=4927, out_features=2785),
            nn.ReLU(),
            nn.Dropout(p=p)
        )
        # 3st dense layer.
        self.DENSE3 = nn.Sequential(
            nn.Linear(in_features=2785, out_features=1574),
            nn.ReLU(),
            nn.Dropout(p=p)
        )
        # FCN layer
        self.FCN = nn.Linear(in_features=1574, out_features=num_classes)

    def forward(self, signal):
        x = self.CNN1(signal.unsqueeze(1))

        x = self.CNN2(x)
        x = torch.flatten(x, -2, -1)
        x = torch.unsqueeze(x, dim=1)
        x = self.DENSE1(x)
        x = self.DENSE2(x)
        x = self.DENSE3(x)
        x = self.FCN(x).squeeze(1)

        return x


def fit(model, data_loader, optimizer, loss_func):
    sum_losses = 0

    model.train()
    for x, y in data_loader:
        preds = model(x.cuda())
        loss = loss_func(preds, y.view(-1, 1).cuda())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        sum_losses += loss.item()

    return sum_losses / len(data_loader)


def predict(model, data_loader):
    model.eval()
    with torch.no_grad():
        return torch.cat([model(x.cuda()) for x, _ in data_loader], dim=0).cpu()
