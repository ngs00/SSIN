import torch
from torch.nn.functional import relu


class FCNN(torch.nn.Module):
    def __init__(self, len_spect, num_classes):
        super(FCNN, self).__init__()
        self.fc1 = torch.nn.Linear(len_spect, 256)
        self.fc2 = torch.nn.Linear(256, 256)
        self.fc3 = torch.nn.Linear(256, num_classes)

    def forward(self, x):
        h = relu(self.fc1(x))
        h = relu(self.fc2(h))
        out = self.fc3(h)

        return out


def fit(model, data_loader, optimizer, loss_func):
    sum_losses = 0

    model.train()
    for x, y in data_loader:
        preds = model(x.cuda())
        loss = loss_func(preds, y.cuda())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        sum_losses += loss.item()

    return sum_losses / len(data_loader)


def predict(model, data_loader):
    model.eval()
    with torch.no_grad():
        return torch.cat([model(x.cuda()) for x, _ in data_loader], dim=0).cpu()
