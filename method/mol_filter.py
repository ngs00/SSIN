import torch
import torch.nn as nn


class MolFilter(nn.Module):
    def __init__(self, signal_size=1024, kernel_size=11, in_ch=1, num_classes=17, p=0.48599073736368):
        super(MolFilter, self).__init__()
        self.num_classes = num_classes
        self.CNN1 = nn.Sequential(
            nn.Conv1d(in_channels=in_ch, out_channels=31, kernel_size=kernel_size, stride=1, padding=0),
            nn.BatchNorm1d(num_features=31),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.cnn1_size = int(((signal_size - kernel_size + 1 - 2) / 2) + 1)
        self.CNN2 = nn.Sequential(
            nn.Conv1d(in_channels=31, out_channels=62, kernel_size=kernel_size, stride=1, padding=0),
            nn.BatchNorm1d(num_features=62),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.cnn2_size = int(((self.cnn1_size - kernel_size + 1 - 2) / 2) + 1)
        self.DENSE1 = nn.Sequential(
            nn.Linear(in_features=24738, out_features=4927),
            nn.ReLU(),
            nn.Dropout(p=p)
        )
        self.DENSE2 = nn.Sequential(
            nn.Linear(in_features=4927, out_features=2785),
            nn.ReLU(),
            nn.Dropout(p=p)
        )
        self.DENSE3 = nn.Sequential(
            nn.Linear(in_features=2785, out_features=256),
            nn.ReLU(),
            nn.Dropout(p=p)
        )
        self.fc_fg = nn.Linear(in_features=256, out_features=num_classes)
        self.fc_c = nn.Linear(in_features=256, out_features=1)
        self.fc_n = nn.Linear(in_features=256, out_features=1)
        self.fc_o = nn.Linear(in_features=256, out_features=1)
        self.fc_mw = nn.Linear(in_features=256, out_features=1)

    def forward(self, signal):
        x = self.CNN1(signal.unsqueeze(1))
        x = self.CNN2(x)
        x = torch.flatten(x, -2, -1)
        x = torch.unsqueeze(x, dim=1)
        x = self.DENSE1(x)
        x = self.DENSE2(x)
        x = self.DENSE3(x)
        y_fg = self.fc_fg(x).squeeze(1)
        y_num_c = self.fc_c(x).squeeze(1)
        y_num_n = self.fc_c(x).squeeze(1)
        y_num_o = self.fc_o(x).squeeze(1)
        y_mw = self.fc_mw(x).squeeze(1)

        return y_fg, y_num_c, y_num_n, y_num_o, y_mw

    def fit(self, data_loader, optimizer, loss_fg, loss_n_c, loss_n_n, loss_n_o, loss_mw):
        sum_losses = 0

        self.train()
        for x, y, num_c, num_n, num_o, mw in data_loader:
            y_fg, y_num_c, y_num_n, y_num_o, y_mw = self(x.cuda())
            _loss_fg = loss_fg(y_fg, y.cuda())
            # _loss_n_c = loss_n_c(y_num_c, num_c.view(-1, 1).cuda())
            # _loss_n_n = loss_n_n(y_num_n, num_n.view(-1, 1).cuda())
            # _loss_n_o = loss_n_o(y_num_o, num_o.view(-1, 1).cuda())
            _loss_mw = loss_mw(y_mw, mw.view(-1, 1).cuda())

            # loss = _loss_fg + _loss_n_c + _loss_n_n + _loss_n_o + _loss_mw
            loss = _loss_fg + _loss_mw

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            sum_losses += loss.item()

        return sum_losses / len(data_loader)

    def predict(self, data_loader):
        list_fg = list()
        list_num_c = list()
        list_num_n = list()
        list_num_o = list()
        list_mw = list()

        self.eval()
        with torch.no_grad():
            for x, _, _, _, _, _ in data_loader:
                y_fg, y_num_c, y_num_n, y_num_o, y_mw = self(x.cuda())
                list_fg.append(y_fg)
                list_num_c.append(y_num_c)
                list_num_n.append(y_num_n)
                list_num_o.append(y_num_o)
                list_mw.append(y_mw)
        preds_fg = torch.cat(list_fg, dim=0).cpu()
        preds_num_c = torch.cat(list_num_c, dim=0).cpu()
        preds_num_n = torch.cat(list_num_n, dim=0).cpu()
        preds_num_o = torch.cat(list_num_o, dim=0).cpu()
        preds_mw = torch.cat(list_mw, dim=0).cpu()

        return preds_fg, preds_num_c, preds_num_n, preds_num_o, preds_mw
