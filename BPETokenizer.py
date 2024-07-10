from utils import get_pairs, merge

class BPETokenizer():
    def __init__(self):
        # used for encoding, stores all the merged tokens
        self.merges = {}
        # used for decoding, maps the tokens to their utf-8 encoded bytes.
        self.vocab = {}
    
    def train(self, text, vocab_size, verbose=False):

        tokens = list(text.encode('utf-8')) # convert to utf-8 encodings
        
        merges_required = vocab_size - 256  # vocab size is the number of desired tokens. 
        
        # a lookup table that maps the tokens to their appropriate bytes.
        self.vocab = {idx : bytes([idx]) for idx in range(256)}

        for i in range(merges_required):
            all_pairs = get_pairs(tokens) # all consecutive pairs
            idx = 256 + i                   # mint new token
            most_common_pair = max(all_pairs, key = all_pairs.get) # find the pair with most common occurence

            tokens = merge(tokens,most_common_pair, idx) # merge the pair, replacing it with the minted token
            if verbose:
                print(f'Merging {most_common_pair} to token : {idx}')
            
            self.merges[most_common_pair] = idx
            
            # vocab for merges, will use existing bytes to merge
            self.vocab[idx] = self.vocab[most_common_pair[0]] + self.vocab[most_common_pair[1]]


    def encode(self, text):

        tokens = list(text.encode('utf-8'))

        while len(tokens) >= 2:
            # returns all pairs with their counts. 
            all_pairs = get_pairs(tokens)
            
            # returns the pair with least merge index. if pair isnt found in merges, return inf.
            pair = min(all_pairs, key = lambda p: self.merges.get(p, float('inf')))

            if pair not in self.merges:
                break
            tokens = merge(tokens, pair, self.merges[pair])

        return tokens
    
    def decode(self, token_ids):
        
        # use the vocab to get bytes, decode using utf-8

        text = b"".join(self.vocab[id] for id in token_ids)
        text = text.decode('utf-8', errors='replace')
        
        return text
    

tokenizer = BPETokenizer()

train_text = open('taylorswift.txt').read()

tokenizer.train(train_text,512, verbose=True)

print(tokenizer.decode(tokenizer.encode('hello world!')))