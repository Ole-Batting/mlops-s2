import torch
import torchvision.transforms as transforms
from model_lightning import MyAwesomeModel
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

dataset_path = "datasets"
cuda = torch.cuda.is_available()
DEVICE = torch.device("cuda" if cuda else "cpu")
batch_size = 100

mnist_transform = transforms.Compose([transforms.ToTensor()])

train_dataset = MNIST(
    dataset_path, transform=mnist_transform, train=True, download=True
)
test_dataset = MNIST(
    dataset_path, transform=mnist_transform, train=False, download=True
)

train_loader = DataLoader(
    dataset=train_dataset, batch_size=batch_size, shuffle=True
)
test_loader = DataLoader(
    dataset=test_dataset, batch_size=batch_size, shuffle=False
)

model = MyAwesomeModel()
trainer = Trainer(max_epochs=20, limit_train_batches=0.2)
trainer.fit(model, train_loader)
