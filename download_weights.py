import os

import gdown

from src.utils.text_encoder_utils import download_lm


def download():
    gdown.download(id="1cEHGhEktlg_gCRKdevmKOoCJIumscE5v")  # download model_best.pth
    gdown.download(
        id="1xhflKMPB6QBQpwnZC2sSaHnSpj-64bZf"
    )  # download librispeech-vocab.txt
    gdown.download(
        id="1K2LIfCCxiSQF5xx0pUPS9qQpVEu8qJCT"
    )  # download tokenizer-wiki.json

    download_lm()

    os.rename("model_best.pth", "data/other/model_best.pth")
    os.rename("librispeech-vocab.txt", "data/other/librispeech-vocab.txt")
    os.rename("tokenizer-wiki.json", "data/other/tokenizer-wiki.json")


if __name__ == "__main__":
    download()
