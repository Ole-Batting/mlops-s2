import sys

import torch


def main(model_filepath, input_filepath):

    model = torch.load(model_filepath)
    data = torch.load(input_filepath)
    with torch.no_grad():
        model.eval()
        log_ps = model(data)
        _, top_class = log_ps.topk(1, dim=1)
        print(top_class)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
