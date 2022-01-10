import sys
sys.path.append('src/models')
from train_model import _main

try:
    _main('data/processed', 'models/model.pth', 1, True)
    val = True
except:
    val = False

assert val, 'training failed on 1 epoch.'
