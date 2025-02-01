import regex as re

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
            stats = {}
            for j in range(len(tokens)):
                get_stats(tokens[j], stats)
            pair_to_merge = max(stats, key=stats.get)
            self.merges[pair_to_merge] = 256 + i
            for j in range(len(tokens)):
                tokens[j] = merge(tokens[j], pair_to_merge, 256 + i)
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
            stats = get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            ind = self.merges[pair]
            ids = merge(ids, pair, ind)
        return ids


    def decode(self, ids):
        tokens = b"".join(self.vocab[idx] for idx in ids)
        text = tokens.decode("utf-8", errors="replace")
        return text


tok = RegexTokenizer()
text = open("taylorswift.txt", 'r').read()
print(len(list(set(text))))
vocab_size = 30 + 256
tok.train(text, vocab_size)
test = "09 premiere of Hannah Montana: The Movie. She had a cameo appearance in the film and wrote two songs for its soundtrack.[71][72]Swift's second studio album, Fearless, was"
print(tok.decode(tok.encode(test)) == test)