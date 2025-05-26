import torch


class CNN(torch.nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=4, stride=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=4),
            torch.nn.Dropout(p=0.2)
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=4),
            torch.nn.Dropout(p=0.2)
        )
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(16896, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.2),
            torch.nn.Linear(128, num_classes)
        )

    def forward(self, x):
        h = self.conv1(x)
        h = self.conv2(h).view(x.shape[0], -1)
        out = self.fc(h)

        return out


def fit(model, data_loader, optimizer, loss_func):
    sum_losses = 0

    model.train()
    for x, y in data_loader:
        optimizer.zero_grad()

        preds = model(x.cuda())
        loss = loss_func(preds, y.cuda())

        loss.backward()
        optimizer.step()
        sum_losses += loss.item()

    return sum_losses / len(data_loader)


def predict(model, data_loader):
    model.eval()
    with torch.no_grad():
        return torch.cat([model(x.cuda()) for x, _ in data_loader], dim=0).cpu()
