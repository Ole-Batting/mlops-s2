import torch
import os.path
import pytest

train_path = "data/processed/train.pth"
test_path = "data/processed/test.pth"
cond = os.path.exists(train_path) and not os.path.exists(test_path)


@pytest.mark.skipif(not cond, reason="Data files not found")
def test_data():
    train_set = torch.load(train_path)
    test_set = torch.load(test_path)

    for images, labels in train_set:
        assert all(
            [images.size(dim=i) == j for i, j in zip(range(4), [5000, 1, 28, 28])]
        ), "incorrect train size."
        assert all(
            [i in labels for i in range(10)]
        ), "not all labels are represented in train."

    assert all(
        [test_set[0].size(dim=i) == j for i, j in zip(range(4), [5000, 1, 28, 28])]
    ), "incorrect test size."
    assert all(
        [i in test_set[1] for i in range(10)]
    ), "not all labels are represented in test."
