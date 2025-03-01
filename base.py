"""
This file contains basic Tokenizer class
and some functions which will be used in later
implementations of Tokenizer.
"""

def get_stats(ids, counts=None):
    if counts is None:
        counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge(ids, pair, ind):
    newids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and pair[0] == ids[i] and pair[1] == ids[i + 1]:
            newids.append(ind)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids

class Tokenizer:
    def __init__(self):
        self.merges = {} # (int, int) -> int
        """
        Merges contains the history of merges.
        If merges[(1,2)] equals 3 it means that pair of tokens (1,2)
        was replaced by token 3.
        """
        self.vocab = {} # int -> bytes
        """
        Vocab is reversed merges, but tokens are concatenated as bytes
        If merges[(1,2)] equals 3, vocab[3] equals bytes(1) + bytes(2)
        """

        self.pattern = ""

    #bacis methods

    def train(self, text, vocab_size):
        raise NotImplementedError

    def encode(self, text):
        raise NotImplementedError

    def decode(self, ids):
        raise NotImplementedError

    def create_vocab(self):
        for i in range(256):
            self.vocab[i] = bytes([i])
        for (p0, p1), ind in self.merges.items():
            self.vocab[ind] = self.vocab[p0] + self.vocab[p1]

