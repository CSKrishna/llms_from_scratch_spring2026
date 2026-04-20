
import base64
from collections import Counter
import json
from pathlib import Path
import time

import regex as re
from cs336_basics.utils import PATTERN
from cs336_basics.dataloaders import ChunkIterator, DocIterator
#from cs336_basics.utils import PATTERN



class BasicTokenizer():
    def __init__(self, special_tokens: list[str] | None = None, vocab_size: int = 256, verbose = True):
       
        # Your list of special tokens; special_tokens = ["<|endoftext|>", "<|startoftext|>"]
        self.special_tokens = special_tokens
        self.vocab_size = vocab_size

        #mapping from token index in vocab to its UTF-8 byte representation
        self.vocab = {idx: bytes([idx]) for idx in range(256)} 
        self.idx = 256
        for j in range(len(special_tokens)):
            self.vocab[self.idx] = special_tokens[j].encode("utf-8")
            self.idx += 1

        #list of merged byte-pairs
        self.merges: list[tuple[bytes, bytes]] = []
        self.pattern = PATTERN
        self.pre_tokens: dict[tuple[bytes], int] = Counter()
        self.byte_pairs: dict[tuple[bytes, bytes], int] = Counter()
        self.verbose = verbose

       

    def _count_byte_pairs(self, b: tuple[bytes]):
        for pair1, pair2 in zip(b, b[1:]):
            self.byte_pairs[(pair1, pair2)] += 1
    


    def _init_structures(self,input_path: str):
        """
         Chunk the file into parts that can be counted independently.
         For each chunk, materialize the set of pre-tokens, initialize byte_pairs, pre_token counts.
        """
        chunk_iterator = ChunkIterator(input_path)
        for chunk in chunk_iterator:
            di = DocIterator(chunk, self.special_tokens)
            for doc in di:
                pre_tokens = re.findall(self.pattern, doc)
                for pre_token in pre_tokens:
                    pt_b = pre_token.encode("utf-8")
                    pt_b = tuple(bytes([pt_b[i]]) for i in range(len(pt_b)))
                    self.pre_tokens[pt_b] += 1
                    self._count_byte_pairs(pt_b)

    def _merge(self, pre_token: tuple[bytes], pair: tuple[bytes, bytes])-> tuple[bytes]:
        merged_pre_token = [] #list of byte objects, later converted to a tuple of bytes
        i = 0
        l = len(pre_token) - 1
        while i < l:
            if (pre_token[i], pre_token[i+1]) == pair:
                merged_pre_token.append(pre_token[i] + pre_token[i+1])
                #adjust byte-pair counts for neighbours
                if i > 0:
                    key = (pre_token[i-1], pre_token[i]) 
                    self._adjust_byte_pairs(key, pre_token)  
                    key1 =  (pre_token[i-1], pre_token[i] + pre_token[i+1]) 
                    self.byte_pairs[key1] += self.pre_tokens[pre_token]
                if i < l -1:
                    key = (pre_token[i+1],pre_token[i+2]) 
                    self._adjust_byte_pairs(key, pre_token) 
                    key1 =  (pre_token[i] + pre_token[i+1], pre_token[i+2]) 
                    self.byte_pairs[key1] += self.pre_tokens[pre_token]
                i += 2
            else:
                merged_pre_token.append(pre_token[i])
                i += 1
        if i == l:
            merged_pre_token.append(pre_token[i])  
        return tuple(merged_pre_token)
    
  
    
    def _adjust_byte_pairs(self, key, pre_token):
        if key in self.byte_pairs:
            count = self.byte_pairs[key] - self.pre_tokens[pre_token]
            if count > 0: self.byte_pairs[key] = count
            else: del self.byte_pairs[key]  
        


    def train(self, input_path: str):
        l = 0
        if self.special_tokens:
            l = len(self.special_tokens)
        num_merges = self.vocab_size - l - 256
        if num_merges <= 0: return None

        self._init_structures(input_path)

        for i in range(num_merges):
            best_key = max(self.byte_pairs, key=lambda k: (self.byte_pairs[k], k))
            for pre_token in list(self.pre_tokens):
                merged_pre_token = self._merge(pre_token, best_key)
                if merged_pre_token != pre_token:
                    self.pre_tokens[merged_pre_token] = self.pre_tokens[pre_token]
                    del self.pre_tokens[pre_token]     
           
            self.vocab[self.idx] = best_key[0] + best_key[1] 
            self.merges.append(best_key)
            if self.verbose:
                print(f"merge {i+1}/{num_merges}: {best_key} -> {self.idx} ({self.vocab[self.idx]}) had {self.byte_pairs[best_key]} occurrences")
            del self.byte_pairs[best_key] 
            self.idx += 1
        return self.vocab, self.merges
    
    def _save_vocab(self, path: str | Path):
        n = len(self.vocab)
        ordered = [self.vocab[i] for i in range(n)]
        as_b64 = [base64.b64encode(b).decode("ascii") for b in ordered]
        Path(path).write_text(json.dumps(as_b64), encoding="utf-8")
    
    def _save_merges(self, path: str | Path):
        payload = [
             [
                 base64.b64encode(left).decode("ascii"),
                 base64.b64encode(right).decode("ascii"),
             ]
        for left, right in self.merges]
        Path(path).write_text(json.dumps(payload), encoding="utf-8")
        
    



if __name__ == '__main__':
    input_path = "data/TinyStoriesV2-GPT4-valid.txt"
    special_tokens = ["<|endoftext|>"]
    bt = BasicTokenizer(special_tokens, 10000, False)
    t0 = time.time()
    vocab, merge = bt.train(input_path)
    t1 = time.time()
    print(f"Training took {t1 - t0:.2f} seconds")
    

    
   




        
