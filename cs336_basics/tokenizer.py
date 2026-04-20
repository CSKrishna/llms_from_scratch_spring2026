from collections import Counter
from typing import Iterable, Iterator
import regex as re
from .train_bpe import BasicTokenizer
from cs336_basics.utils import str_to_bytes, find_chunk_boundaries
#from .dataloaders import PreTokenIterator
from cs336_basics.dataloaders import ChunkIterator
import base64
import json
from pathlib import Path


PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


class Tokenizer():
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] = ["<|endoftext|>"], pattern = PATTERN):
        self.vocab: dict[int, bytes] = vocab
        self.special_tokens = special_tokens
        self.pattern = pattern
        self.merges_dict = {tup[0] + tup[1]: i + 256 for i, tup in enumerate(merges)} #map byte-pair to token index offset by 256
        for i in range(256):
            self.merges_dict[self.vocab[i]] = i
        self.base_size = len(self.merges_dict)
        if special_tokens:
            self.ordered = sorted(special_tokens, key=len, reverse=True)
            self.split_pattern = "|".join(map(re.escape, self.ordered))

    def decode(self, ids: list[int]) -> str:
        if not ids: return ""
        b = b"".join(self.vocab[i] for i in ids)
        return b.decode("utf-8", errors="replace")

    
    def encode(self, text: str) -> list[int]:
        l = []
        if self.special_tokens:
            parts = re.split(f"({self.split_pattern})", text)
            for part in parts:
                if part in self.special_tokens:
                    l.extend([self.special_tokens.index(part) + self.base_size])
                else: self._encode(part, l)
        else:
            self._encode(text, l)
        return l


    
    def _encode(self, text: str, l: list[int]):
        pre_tokens = re.findall(self.pattern, text)
        for pre_token in pre_tokens:
            pt_b = str_to_bytes(pre_token)
            #counts = self._count_byte_pairs(pt_b)
            while True:
                counts = self._count_byte_pairs(pt_b)
                if len(counts) < 1: break
                pair = min(counts, key=lambda p: self.merges_dict.get(p[0] + p[1], float("inf")))
                merged = pair[0] + pair[1]
                if merged not in self.merges_dict: break
                pt_b = self._merge(pt_b, pair)
            l.extend([self.merges_dict[item] for item in pt_b])
            
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        ci = ChunkIterator.from_file(iterable)
        for chunk in ci:
            documents = iter(re.split(f"({self.split_pattern})", chunk))
            for doc in documents:
                if doc in self.special_tokens:
                    yield self.special_tokens.index(doc) + self.base_size
                else:
                    l = []
                    self._encode(doc,l)
                    yield from l
    
    def _count_byte_pairs(self, b: tuple[bytes]) -> dict[tuple[bytes, bytes], int]:
        counts = Counter()
        for pair1, pair2 in zip(b, b[1:]):
            counts[(pair1, pair2)] += 1
        return counts
    
    def _merge(self, pre_token: tuple[bytes], pair: tuple[bytes, bytes])-> tuple[bytes]:
        merged_pre_token: list[bytes] = []#list of byte objects, later converted to a tuple of bytes
        i = 0
        l = len(pre_token) - 1
        while i < l:
            if (pre_token[i], pre_token[i+1]) == pair:
                merged_pre_token.append(pre_token[i] + pre_token[i+1])
                key = (pre_token[i], pre_token[i+1]) 
                i += 2
            else:
                merged_pre_token.append(pre_token[i])
                i += 1
        if i == l:
            merged_pre_token.append(pre_token[i])  
        return tuple(merged_pre_token)
    
    @staticmethod
    def _load_vocab_from_save(path: str | Path) -> dict[int, bytes]:
        text = Path(path).read_text(encoding="utf-8")
        as_b64: list[str] = json.loads(text)
        return {i: base64.b64decode(s) for i, s in enumerate(as_b64)}

    @staticmethod
    def _load_merges_from_file(path: str | Path) -> list[tuple[bytes, bytes]]:
        text = Path(path).read_text(encoding="utf-8")
        payload: list[list[str]] = json.loads(text)
        merges: list[tuple[bytes, bytes]] = []
        for pair in payload:
            if len(pair) != 2:
                raise ValueError(f"expected [left_b64, right_b64], got {pair!r}")
            left_b64, right_b64 = pair
            merges.append((base64.b64decode(left_b64), base64.b64decode(right_b64)))
        return merges

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str | Path,
        merges_filepath: str | Path,
        special_tokens: list[str] | None = None,
        pattern: str = PATTERN,
    ) -> "Tokenizer":
        if special_tokens is None:
            special_tokens = ["<|endoftext|>"]
        vocab = cls._load_vocab_from_save(vocab_filepath)
        merges = cls._load_merges_from_file(merges_filepath)
        return cls(vocab, merges, special_tokens=special_tokens, pattern=pattern)
    


if __name__ == '__main__':
    input_path = "data/TinyStoriesV2-GPT4-valid.txt"
    special_tokens = ["<|endoftext|>"]
    bt = BasicTokenizer(special_tokens, 2000, False)
    vocab, merges = bt.train(input_path)
    tokenizer = Tokenizer(vocab, merges, special_tokens)
    text = "he is a good guy"
    l = []
    tokenizer._encode(text, l)
    print(l)
    

    
