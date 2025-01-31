import regex as re

def get_stats(ids: [list[list[int]]]):
    res = {}
    for lst in ids:
        for pair in zip(lst, lst[1:]):
            res[pair] = res.get(pair, 0) + 1
    return res

def merge(ids: list[list[int]], pair, ind: int):
    newids = []
    for lst in ids:
        newlst = []
        i = 0
        while i < len(lst):
            if i < len(lst) - 1 and pair[0] == lst[i] and pair[1] == lst[i+1]:
                newlst.append(ind)
                i += 2
            else:
                newlst.append(lst[i])
                i += 1

        newids.append(newlst)
    return newids


class RegexTokenizer:
    def __init__(self):
        self.merges = {}
        self.vocab = {}
        self.pattern  = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

    def train(self, text, vocab_size, verbose=False):
        splitted_text: list[str] = re.findall(self.pattern, text)
        tokens: list[list[int]] = [list(map(int, word.encode("utf-8"))) for word in splitted_text]
        i = 0
        num_merges = vocab_size - 256
        for i in range(num_merges):
            stats = get_stats(tokens)
            pair_to_merge = max(stats, key=stats.get)
            self.merges[pair_to_merge] = 256 + i
            tokens = merge(tokens, pair_to_merge, 256 + i)
            print(f"merged pair {pair_to_merge} to token {256 + i}")
        self.create_vocab()

    def create_vocab(self):
        for i in range(256):
            self.vocab[i] = bytes([i])
        for (p0, p1), ind in self.merges.items():
            self.vocab[ind] = self.vocab[p0] + self.vocab[p1]

    def encode(self, text):
        ids = list(map(int, text.encode("utf-8")))
        while len(ids) >=2:
            stats = get_stats([ids])
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            ind = self.merges[pair]
            ids = merge([ids], pair, ind)
        return ids[0]


    def decode(self, ids):
        tokens = b"".join(self.vocab[idx] for idx in [lst for lst in ids])
        text = tokens.decode("utf-8", errors="replace")
        return text


