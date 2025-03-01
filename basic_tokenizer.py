from base import Tokenizer, get_stats, merge

class BasicTokenizer(Tokenizer):

    def __init__(self):
        super().__init__()

    def train(self, text, vocab_size, print_logs=False):
        assert vocab_size >= 256
        num_merges = vocab_size - 256

        ids = list(text.encode('utf-8'))
        for i in range(num_merges):
            stats = get_stats(ids)
            pair_to_merge = max(stats, key = stats.get)
            idx = 256 + i
            ids = merge(ids, pair_to_merge, idx)
            self.merges[pair_to_merge] = idx
            if print_logs:
                print(f"merged tokens {pair_to_merge} to token {idx}")

        self.create_vocab()

    def encode(self, text):
        ids = list(text.encode('utf-8'))
        while len(ids) >= 2:
            stats = get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            idx = self.merges[pair]
            ids = merge(ids, pair, idx)
        return ids

    def decode(self, ids):
        tokens = b"".join(self.vocab[idx] for idx in ids)
        text = tokens.decode("utf-8", errors="replace")
        return text

