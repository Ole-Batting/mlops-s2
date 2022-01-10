import sys
sys.path.append('src/models')
from train_model import _main
import numpy as np
import torch

x = torch.Tensor(np.ones((10,1,28,28)))
y = torch.tensor(np.arange((10)),dtype=torch.long)

val = False
try:
    _main([[x, y]], [x, y], 'models/model.pth', 1, True)
    val = True

assert val, 'training failed on 1 epoch.'
