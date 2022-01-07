from pytorch_lightning import LightningModule
from torch import nn, optim

class MyAwesomeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 64, 3),
            nn.LeakyReLU(),
            nn.Conv2d(64, 32, 3),
            nn.LeakyReLU(),
            nn.Conv2d(32, 16, 3),
            nn.LeakyReLU(),
            nn.Conv2d(16, 8, 3),
            nn.LeakyReLU()
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(8 * 20 * 20, 128),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(120, 10)
        )
        self.criterium = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.classifier(self.backbone(x))

    def training_step(self, batch, batch_idx):
        data, target = batch
        preds = self(data)
        loss = self.criterium(preds, target)
        return loss
    
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-2)