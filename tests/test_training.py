import sys
sys.path.append('src/models')
from train_model import _main

_main('data/processed', 'models/model.pth', 1, True)
