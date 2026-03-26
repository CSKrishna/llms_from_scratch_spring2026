import os
import regex as re
from cs336_basics.utils import find_chunk_boundaries
from cs336_basics.train_bpe import PATTERN


class ChunkIterator():
    def __init__(self, input_path: str):
        if not os.path.isfile(input_path):
            raise FileNotFoundError(f"Path {input_path} does not exist.")
        
        self.f = open(input_path, "rb")
        num_processes = 4
        self.boundaries = find_chunk_boundaries(self.f, num_processes, b"<|endoftext|>")
        self.internal_zip = zip(self.boundaries[:-1], self.boundaries[1:])
     

    def __iter__(self):
        return self

    def __next__(self):
        start, end = next(self.internal_zip)
        self.f.seek(start)
        chunk_bytes = self.f.read(end - start)
            
        return chunk_bytes.decode("utf-8", errors="ignore")

#splits chunk into documents based on document delimiters
class DocIterator():
    def __init__(self, chunk: str, special_tokens):
        self.special_tokens = special_tokens
        split_pattern = "|".join(map(re.escape, self.special_tokens))
        self.documents = iter(re.split(f"({split_pattern})", chunk))

    def __iter__(self):
        return self
    
    def __next__(self):
        doc = next(self.documents)
        while doc in self.special_tokens:
            doc = next(self.documents)
        return doc 
    

    
        
class PreTokenIterator:
    def __init__(self, file_path, special_tokens, pattern = PATTERN):
        self.chunk_iterator = ChunkIterator(file_path)
        self.special_tokens = set(special_tokens)
        self.pattern = pattern
        self.split_pattern = "|".join(map(re.escape, self.special_tokens))
        self._generator_state = self._generate_tokens()

    def _generate_tokens(self):
        """The logic for nesting is handled here via yield."""
        for chunk in self.chunk_iterator:
            docs = re.split(f"({self.split_pattern})", chunk)
            for doc in docs:
                if doc in self.special_tokens:
                    yield doc
                else:
                    pre_tokens = re.findall(self.pattern, doc)
                    for token in pre_tokens:
                        yield token

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._generator_state)
            
            
if __name__ == '__main__':
    input_path = "data/TinyStoriesV2-GPT4-valid.txt"
    special_tokens = ["<|endoftext|>"]
    pt = PreTokenIterator(input_path, special_tokens)
    i = 0
    for pts in pt:
        if i > 20: break
        print(pts)
        i += 1
  

