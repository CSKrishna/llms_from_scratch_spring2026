from collections import Counter
from typing import Iterable, Iterator
import regex as re
from .train_bpe import BasicTokenizer
from .utils import str_to_bytes
from .dataloaders import PreTokenIterator


PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


class Tokenizer():
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None, pattern = PATTERN):
        self.vocab: dict[int, bytes] = vocab
        self.special_tokens = special_tokens
        self.pattern =  pattern
        self.merges_dict = {tup[0] + tup[1]: i for i, tup in enumerate(merges)} #map byte-pair to token index offset by 256
        for i in range(256):
            self.merges_dict[self.vocab[i]] = i
        #self.merges_dict.update((val, i) for i, val in enumerate(self.vocab[:256]))
        self.base_size = len(vocab)

    #def from_files(cls, vocab_filepath, merges_filepath, special_tokens: list[str] | None = None):
   
    def decode(self, ids: list[int]) -> str:
        if not ids: return ""
        b = self.vocab[ids[0]]
        for indx in ids[1:]:
            b += self.vocab[indx]
        return b.decode("utf-8", errors="replace")

    def encode1(self, text: str) -> list[int]:
        """
        Encode an input text into a sequence of token IDs
        """
        self.pre_token_iter = PreTokenIterator(text, self.special_tokens, self.pattern)
        l = []
        for doc in self.pre_token_iter:
            if doc in self.special_tokens:
                l.extend(self.special_tokens.index(doc) + self.base_size)
            else:
                self._encode(text, l)
        return l
    
    def encode(self, text: str) -> list[int]:
        l = []
        pre_tokens = re.findall(self.pattern, text)
        for pre_token in pre_tokens:
            if self.special_tokens and pre_token in  self.special_tokens:
                l.extend(self.special_tokens.index(pre_token) + self.base_size)
            else: self._encode(text, l)
        return l


    
    def _encode(self, text: str, l: list[int]):
        pre_tokens = re.findall(self.pattern, text)
        for pre_token in pre_tokens:
            pt_b = str_to_bytes(pre_token)
            counts = self._count_byte_pairs(pt_b)
            while len(counts) > 0:
                pair = min(counts, key=lambda p: self.merges_dict.get(p, float("inf")))
                if pair[0] + pair[1] in self.merges_dict: pt_b = self._merge(pt_b, pair, counts)
                else: break
            pt_b_i = [self.merges_dict[item] for item in pt_b]
            l.extend(pt_b_i)
            
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for st in iterable:
            if st in self.special_tokens:
                yield self.special_tokens.index(st) + self.base_size
            else:
                l = []
                self._encode(st,l)
                yield next(iter(l))

    def _get_index(self, b: bytes) -> int:
        return self.merges_dict[b]
        #if b in self.merges_dict: return self.merges_dict[b] + 256
        #return b[0]
    
    def _count_byte_pairs(self, b: tuple[bytes]) -> dict[tuple[bytes, bytes], int]:
        counts = Counter()
        for pair1, pair2 in zip(b, b[1:]):
            counts[(pair1, pair2)] += 1
        return counts
    
    def _merge(self, pre_token: tuple[bytes], pair: tuple[bytes, bytes], counts: dict[tuple[bytes, bytes], int])-> tuple[bytes]:
        merged_pre_token = [] #list of byte objects, later converted to a tuple of bytes
        i = 0
        l = len(pre_token) - 1
        while i < l:
            if (pre_token[i], pre_token[i+1]) == pair:
                merged_pre_token.append(pre_token[i] + pre_token[i+1])
                key = (pre_token[i], pre_token[i+1]) 
                counts.pop(key, None) 
                #adjust byte-pair counts for neighbours
                if i > 0:
                    key = (pre_token[i-1],pre_token[i]) 
                    counts.pop(key, None) 
                    key1 =  (pre_token[i-1], pre_token[i] + pre_token[i+1]) 
                    counts[key1] += 1
                if i < l -1:
                    key = (pre_token[i+1],pre_token[i+2]) 
                    counts.pop(key, None) 
                    key1 =  (pre_token[i] + pre_token[i+1], pre_token[i+2]) 
                    counts[key1] += 1
                i += 2
            else:
                merged_pre_token.append(pre_token[i])
                i += 1
        if i == l:
            merged_pre_token.append(pre_token[i])  
        return tuple(merged_pre_token)
    


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
    
"""
more efficient implementation of counts
while True:
            counts = self._count_byte_pairs(pt_b)
            if not counts: break
            
            # Find the highest priority merge (lowest rank)
            pair = min(counts, key=lambda p: self.merge_ranks.get(p, float("inf")))
            
            if pair not in self.merge_ranks:
                break
                
            pt_b = self.merge(pt_b, pair)
            
        # Convert the resulting units to IDs
        l.extend(self.vocab[unit] for unit in pt_b)
"""