import regex as re
from utils import get_pairs, merge

class GPTTokenizer():
    def __init__(self, regex_pattern, special_tokens = None):
        self.merges = {}
        self.vocab = {}
        # GPT2 introduced regex pattern matching, to split words and punctuations, to promote tokens merging 
        # into complete words, rather than letters/words with punctuations. (which was a common thing with BPE, it 
        # almost always merged 'e' and ' ')

        self.gptsplitpattern = re.compile(regex_pattern)
        
        # special tokens are tokens different from the usual prompt, or text. They sort of guide/tell the LLM
        # about formatting of text, end-beginning of text/prompt, and other specialised tokens.  
        self.special_tokens = special_tokens
        # converts str->tokenid to tokenid->str
        self.invert_special = {v:k for k,v in self.special_tokens.items()} if special_tokens else None

    def train(self, text, vocab_size, verbose=False):
        text_splits = re.findall(self.gptsplitpattern, text)
        tokens = [list(token.encode('utf-8')) for token in text_splits]

        merges_required = vocab_size - 256

        self.vocab = {idx : bytes([idx]) for idx in range(256)}

        for i in range(merges_required):
            all_pairs = {}
            for chunks in tokens:
                all_pairs.update(get_pairs(chunks))

            idx = 256 + i
            most_common_pair = max(all_pairs, key = all_pairs.get)

            tokens = [merge(chunks,most_common_pair, idx) for chunks in tokens]
            if verbose:
                print(f'Merging {most_common_pair} to token : {idx}')
            
            self.merges[most_common_pair] = idx
            # no need to train/store special tokens on text. they are already stored.
            self.vocab[idx] = self.vocab[most_common_pair[0]] + self.vocab[most_common_pair[1]]
    
    # GPT tokenizer splits text into words, punctuations etc, rather than passing complete string
    # therefore, we do the normal BPE algorithm on the chunks of the text, then concat the results
    # to the final encoded token list.
    def encode_normal(self,text):
        encoded_tokens = []
        
        tokens = [list(token.encode('utf-8')) for token in re.findall(self.gptsplitpattern,text)] 
        for chunk in tokens:
            while len(chunk) >= 2:
                all_pairs = get_pairs(chunk)
                pair = min(all_pairs, key = lambda p: self.merges.get(p, float('inf')))
                if pair not in self.merges:
                    break
                chunk = merge(chunk, pair, self.merges[pair])
            
            encoded_tokens.extend(chunk)
        
        return encoded_tokens
    
    # This is the main encode function, it handles special tokens, by creating a regex pattern to match and split
    # the text into normal text and special tokens.
    def encode(self, text, special_allowed=False):
        # final tokens
        encoded_tokens = []
        
        # check if special tokens allowed
        if special_allowed:
            # creating regex pattern 
            # example, for <start-of-text> it will be (<start\-of\-text>)
            special_pattern = "(" + "|".join(re.escape(s) for s in self.special_tokens) + ")"
            
            # split the text into special and non special.
            special_split = re.split(special_pattern,text)

            for s in special_split:
                # if special token, use the token given to append.
                if s in self.special_tokens:
                    encoded_tokens.append(self.special_tokens[s])
                # if normal, use BPE.
                else:
                    encoded_tokens.extend(self.encode_normal(s))
       
        # If special tokens not allowed, use BPE.
        else:
            encoded_tokens.extend(self.encode_normal(text))
    
    def decode(self, token_ids, special_allowed=False):
        # store array of bytes of chunks of text. (because of the regex chunking)
        byte_chunks = []

        for id in token_ids:
            if id in self.invert_special and special_allowed:
                # if special token encountered, use inverted table to get the str, and encode to get bytes.
                byte_chunks.append(self.invert_special[id].encode('utf-8'))           
            else:
                # else use the vocab.
                byte_chunks.append(self.vocab[id])
        # join all the bytes
        text = b"".join(byte_chunks)
        # decode to utf-8 to get final text.
        return text.decode('utf-8', errors='replace')        

special_tokens = {
    "<start-of-text>" : 696969,
    "<end-of-text>" : 420420,
    "<end-of-prompt>": 42069
}

GPT4Pattern = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
GPT2Pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

tokenizer = GPTTokenizer(GPT4Pattern, special_tokens)

train_text = open('taylorswift.txt').read()

tokenizer.train(train_text, 15120)

test_text = '<start-of-text> hello how are you kaise ho?<end-of-prompt>'

print(tokenizer.decode(tokenizer.encode(test_text, special_allowed=True), special_allowed=True))