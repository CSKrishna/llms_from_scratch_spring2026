import os
import re
from .pretokenization_example import find_chunk_boundaries


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