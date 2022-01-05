from torch import nn


class MyAwesomeModel(nn.Module):
    def __init__(self):
        super().__init__()
        cc = [1, 6, 16]
        ck = [5, 5]
        mk = 2
        ms = 2
        self.conv = nn.Sequential(
            nn.Conv2d(cc[0], cc[1], ck[0]),
            nn.MaxPool2d(mk, ms),
            nn.Conv2d(cc[1], cc[2], ck[0]),
        )

        conv_dim_out_1 = 28 - (ck[0] - 1)  # 28-(5-1)=24
        maxp_dim_out = (
            conv_dim_out_1 - (mk - 1) - 1
        ) // ms + 1  # (24-(2-1)-1)//2+1=22//2+1=12
        conv_dim_out_2 = maxp_dim_out - (ck[1] - 1)  # 12-(5-1)=8

        self.fc = nn.Sequential(
            nn.Linear(cc[-1] * conv_dim_out_2 * conv_dim_out_2, 120),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(84, 10),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, x):
        n = x.shape[0]
        x = self.conv(x)
        x = self.fc(x.view(n, -1))
        return x
