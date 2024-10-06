import gzip
import os
import shutil

import kenlm
import wget
from pyctcdecode import LanguageModel


def get_lm(alpha=0.5, beta=1.5):
    """
    Initialize language model for CTC decoder with beam search

    Args:
        alpha (float): alpha parameter for language model
        beta (float): beta parameter for language model
    Returns:
        language_model (LanguageModel)
    """
    kenlm_model = kenlm.Model("data/other/lowercase_3-gram.pruned.1e-7.arpa")
    with open("data/other/librispeech-vocab.txt") as f:
        unigrams = [t.lower() for t in f.read().strip().split("\n")]
    language_model = LanguageModel(
        kenlm_model,
        unigrams,
        alpha=alpha,
        beta=beta,
        unk_score_offset=-10.0,
        score_boundary=True,
    )
    return language_model


def download_lm():
    if os.path.exists("data/other/lowercase_3-gram.pruned.1e-7.arp"):
        return
    print("Downloading pruned 3-gram model.")
    lm_url = "http://www.openslr.org/resources/11/3-gram.pruned.1e-7.arpa.gz"
    lm_gzip_path = wget.download(lm_url)
    uppercase_lm_path = "3-gram.pruned.1e-7.arpa"
    if not os.path.exists(uppercase_lm_path):
        with gzip.open(lm_gzip_path, "rb") as f_zipped:
            with open(uppercase_lm_path, "wb") as f_unzipped:
                shutil.copyfileobj(f_zipped, f_unzipped)
    lm_path = "lowercase_3-gram.pruned.1e-7.arpa"
    if not os.path.exists(lm_path):
        with open(uppercase_lm_path, "r") as f_upper:
            with open(lm_path, "w") as f_lower:
                for line in f_upper:
                    f_lower.write(line.lower())
    os.remove("3-gram.pruned.1e-7.arpa.gz")
    os.remove("3-gram.pruned.1e-7.arpa")
    os.makedirs("data/other/", exist_ok=True)
    os.rename(
        "lowercase_3-gram.pruned.1e-7.arpa",
        "data/other/lowercase_3-gram.pruned.1e-7.arp",
    )
    print("Downloaded the 3-gram language model.")
