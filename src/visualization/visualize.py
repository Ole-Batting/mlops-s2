import sys

import matplotlib.pyplot as plt
import torch
from sklearn.decomposition import PCA


def main(model_filepath, input_filepath):

    model = torch.load(model_filepath)
    images, labels = torch.load(input_filepath)
    print([10], "load")

    with torch.no_grad():
        model.eval()
        k = 100
        perm = torch.randperm(images.shape[0])
        idx = perm[:k]
        inter_rep = model.conv(images[idx])
        inter_lab = labels[idx]
        print([19], "inter load")
        inter_pca = PCA(n_components=2).fit_transform(inter_rep.view(k, -1))
        print([21], "inter tsne")

        plt.figure(figsize=(4, 4))
        plt.scatter(inter_pca[:, 0], inter_pca[:, 1], c=inter_lab)
        plt.legend()
        plt.savefig("reports/figures/report.png")
        plt.show()


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
