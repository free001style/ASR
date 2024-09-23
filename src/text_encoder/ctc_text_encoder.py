import re
from collections import defaultdict
from string import ascii_lowercase

import numpy as np
import torch
from pyctcdecode import Alphabet, BeamSearchDecoderCTC

from src.utils.io_utils import read_json
from src.utils.text_encoder_utils import get_lm


class CTCTextEncoder:
    EMPTY_TOK = ""

    def __init__(self, use_bpe=False, alphabet_path=None, **kwargs):
        """
        Args:
            use_bpe (bool): whether to use bpe tokenizer instead of per letter.
            alphabet_path (str): path to bpe alphabet.
        """

        if use_bpe:
            alphabet = list(read_json(alphabet_path)["model"]["vocab"])
        else:
            alphabet = list(ascii_lowercase + " ")

        self.alphabet = alphabet
        self.vocab = [self.EMPTY_TOK] + list(self.alphabet)

        self.ind2char = dict(enumerate(self.vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}

        lm = get_lm()
        self.decoder = BeamSearchDecoderCTC(Alphabet(self.vocab, use_bpe), lm)

    def __len__(self):
        return len(self.vocab)

    def __getitem__(self, item: int):
        assert type(item) is int
        return self.ind2char[item]

    def encode(self, text) -> torch.Tensor:
        text = self.normalize_text(text)
        try:
            return torch.Tensor([self.char2ind[char] for char in text]).unsqueeze(0)
        except KeyError:
            unknown_chars = set([char for char in text if char not in self.char2ind])
            raise Exception(
                f"Can't encode text '{text}'. Unknown chars: '{' '.join(unknown_chars)}'"
            )

    def decode(self, inds) -> str:
        """
        Raw decoding without CTC.
        Used to validate the CTC decoding implementation.

        Args:
            inds (list): list of tokens.
        Returns:
            raw_text (str): raw text with empty tokens and repetitions.
        """
        return "".join([self.ind2char[int(ind)] for ind in inds]).strip()

    def ctc_decode(self, inds) -> str:
        decoded = []
        last_char_ind = self.char2ind[self.EMPTY_TOK]
        for ind in inds:
            if ind == last_char_ind:
                continue
            elif ind != self.char2ind[self.EMPTY_TOK]:
                decoded.append(self.ind2char[ind])
            last_char_ind = ind
        return "".join(decoded)

    def ctc_beam_search(
        self, log_probs: torch.Tensor or np.array, beam_size=10
    ) -> list[dict[str, float]]:
        """
        Simple CTC beam search.

        Args:
            log_probs (torch.Tensor | np.array): (T x len(vocab)) tensor of token log probability.
            beam_size (int): maximum number of beams at each step in decoding.
        Returns:
            output (list[dict]): list of decoded texts with their probabilities.
        """
        if isinstance(log_probs, torch.Tensor):
            log_probs = log_probs.detach().cpu().numpy()
        probs = np.exp(log_probs)
        dp = {
            ("", self.EMPTY_TOK): 1.0,
        }
        for prob in probs:
            dp = self.__expand_end_merge_path(dp, prob)
            dp = self.__truncate_paths(dp, beam_size)
        dp = [
            {"hypothesis": prefix, "probability": proba.item()}
            for (prefix, _), proba in sorted(dp.items(), key=lambda x: -x[1])
        ]
        return dp

    def ctc_lm_beam_search(
        self, log_probs: torch.Tensor or np.array, beam_size=10
    ) -> list[dict[str, float]]:
        """
        CTC beam search with LM

        Args:
            log_probs (torch.Tensor | np.array): (T x len(vocab)) tensor of token log probability.
            beam_size (int): maximum number of beams at each step in decoding.
        Returns:
            output (list[dict]): list of decoded text (with probability equal 1).
        """
        return [
            {
                "hypothesis": self.decoder.decode(log_probs, beam_size),
                "probability": 1.0,
            }
        ]

    def __expand_end_merge_path(self, dp, next_token_probs):
        new_dp = defaultdict(float)
        for ind, next_token_prob in enumerate(next_token_probs):
            cur_char = self.ind2char[ind]
            for (prefix, last_char), v in dp.items():
                if last_char == cur_char:
                    new_prefix = prefix
                else:
                    if cur_char != self.EMPTY_TOK:
                        new_prefix = prefix + cur_char
                    else:
                        new_prefix = prefix
                new_dp[(new_prefix, cur_char)] += v * next_token_prob
        return new_dp

    @staticmethod
    def __truncate_paths(dp, beam_size=10):
        return dict(sorted(list(dp.items()), key=lambda x: -x[1])[:beam_size])

    @staticmethod
    def normalize_text(text: str):
        text = text.lower()
        text = re.sub(r"[^a-z ]", "", text)
        return text
