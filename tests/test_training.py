import sys
sys.path.append('src/models')
from train_model import _main
import numpy as np
import torch

x = torch.Tensor(np.ones((10,1,28,28)))
y = torch.tensor(np.arange((10)),dtype=torch.long)

try:
    _main([[x, y]], [x, y], 'models/model.pth', 1, True)
    val = True
except RuntimeError as err:
    print(err)
    val = False

assert val, 'training failed on 1 epoch.'
