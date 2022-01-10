# -*- coding: utf-8 -*-
import logging
from pathlib import Path

import click
import numpy as np
import torch
from dotenv import find_dotenv, load_dotenv
import os


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
def main(input_filepath, output_filepath):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")

    train = []
    for file in os.listdir(input_filepath):
        if 'train' not in file: continue
        with np.load(f"{input_filepath}/{file}") as data:

            train.append(
                [
                    torch.Tensor(data["images"]).view(-1, 1, 28, 28),
                    torch.from_numpy(data["labels"]),
                ]
            )

    test = None
    with np.load(f"{input_filepath}/test.npz") as data:
        test = [
            torch.Tensor(data["images"]).view(-1, 1, 28, 28),
            torch.from_numpy(data["labels"]),
        ]

    torch.save(train, f"{output_filepath}/train.pth")
    torch.save(test, f"{output_filepath}/test.pth")
    torch.save(test[0][:10], f"{output_filepath}/predict.pth")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
