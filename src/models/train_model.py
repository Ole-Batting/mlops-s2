import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from model import MyAwesomeModel


def main(input_filepath, model_filepath):

    model = MyAwesomeModel()
    train_set = torch.load(f"{input_filepath}/train.pth")
    test_set = torch.load(f"{input_filepath}/test.pth")

    criterion = torch.nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

    epochs = 30
    train_losses, test_losses = [], []
    for e in range(epochs):
        model.train()
        running_loss = 0
        steps = 0
        for images, labels in train_set:
            optimizer.zero_grad()
            log_ps = model(images)
            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() / images.shape[0]
            steps += 1
        else:
            with torch.no_grad():
                model.eval()
                images, labels = test_set
                log_ps = model(images)
                _, top_class = log_ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy = torch.mean(equals.type(torch.FloatTensor))
                loss = criterion(log_ps, labels) / images.shape[0]
                test_losses.append(loss)
                print(
                    f"{e}: train loss: {running_loss/steps}, test loss: {loss}, test accuracy: {accuracy}"
                )
        train_losses.append(running_loss / steps)

    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label="train")
    plt.plot(test_losses, label="test")
    plt.legend()
    plt.show()

    torch.save(model, model_filepath)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
