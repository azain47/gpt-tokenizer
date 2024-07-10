# Helper Functions

def get_pairs(token_ids):
    counts = {}
    # gives consecutive tokens. tokens gives full list, tokens[1:] gives list without 1st element.  
    for pair in zip(token_ids, token_ids[1:]): 
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge(token_ids, pair, idx):
    # take the tokens_ids, find the common pairs, and replace it with idx, in the whole array.
    new_token_ids = []
    i = 0

    while i < len(token_ids):
        if i < len(token_ids) - 1  and token_ids[i] == pair[0] and token_ids[i+1] == pair[1]:
            new_token_ids.append(idx)
            i+=2    # we replaced 2 tokens with 1
        else:
            new_token_ids.append(token_ids[i])
            i+=1

    return new_token_ids