import kenlm
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
    language_model = LanguageModel(kenlm_model,
                                   unigrams,
                                   alpha=alpha,
                                   beta=beta,
                                   unk_score_offset=-10.0,
                                   score_boundary=True)
    return language_model
