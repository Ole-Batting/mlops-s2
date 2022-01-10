import sys
sys.path.append('src/models')
from model import MyAwesomeModel
import numpy as np
import torch

model = MyAwesomeModel()
X = torch.Tensor(np.ones((1,1,28,28)))
Y = model(X)
assert all([Y.size(dim = i) == j for i, j in zip(range(2), [1, 10])]), 'model output size incorrect.'
