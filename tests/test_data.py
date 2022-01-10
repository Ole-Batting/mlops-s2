import torch

train_set = torch.load(f"data/processed/train.pth")
test_set = torch.load(f"data/processed/test.pth")

for images, labels in train_set:
    assert all([images.size(dim = i) == j  for i,j in zip(range(4), [5000, 1, 28, 28])]), 'incorrect train size.'
    assert all([i in labels for i in range(10)]), 'not all labels are represented in train.'

assert all([test_set[0].size(dim = i) == j for i,j in zip(range(4), [5000, 1, 28, 28])]), 'incorrect test size.'
assert all([i in test_set[1] for i in range(10)]), 'not all labels are represented in test.'
