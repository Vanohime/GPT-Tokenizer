import regex as re
from base import Tokenizer, get_stats, merge

GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

class RegexTokenizer(Tokenizer):

    def __init__(self):
        super().__init__()
        self.pattern = GPT4_SPLIT_PATTERN
        
    def train(self, text, vocab_size, savelogs=False):
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
            if savelogs:
                print(f"merged pair {pair_to_merge} to token {256 + i}")
        self.create_vocab()

    def _encode_chunk(self, text_ints):
        while len(text_ints) >=2:
            stats = get_stats(text_ints)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            ind = self.merges[pair]
            text_ints = merge(text_ints, pair, ind)
        return text_ints

    def encode(self, text):
        splitted_text = re.findall(self.pattern, text)
        int_chunks = [list(map(int, word.encode("utf-8"))) for word in splitted_text]
        ids = []
        for chunk in int_chunks:
            encoded_chunk = self._encode_chunk(chunk)
            ids.extend(encoded_chunk)
        return ids

    def decode(self, ids):
        tokens = b"".join(self.vocab[idx] for idx in ids)
        text = tokens.decode("utf-8", errors="replace")
        return text
