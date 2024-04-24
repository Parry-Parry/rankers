from itertools import chain
import torch

class DotDataCollator:
    def __init__(self, tokenizer, special_mask=False):
        self.tokenizer = tokenizer
        self.q_max_length = 30
        self.d_max_length = 200
        self.special_mask = special_mask

    def __call__(self, batch):
        batch_queries = []
        batch_docs = []
        batch_scores = []
        for (q, dx, *args) in batch:
            batch_queries.append(q)
            batch_docs.extend(dx)
            if len(args) == 0:
                continue
            batch_scores.extend(args[0])

        batch_queries = list(chain.from_iterable(batch_queries))
        print(batch_queries)
        batch_scores = list(chain.from_iterable(batch_scores))

        tokenized_queries = self.tokenizer(
            batch_queries,
            padding=True,
            truncation=True,
            max_length=self.q_max_length,
            return_tensors="pt",
            return_special_tokens_mask=self.special_mask,
        )
        tokenized_docs = self.tokenizer(
            batch_docs,
            padding=True,
            truncation=True,
            max_length=self.d_max_length,
            return_tensors="pt",
            return_special_tokens_mask=self.special_mask
        )
        return {
            "queries": dict(tokenized_queries),
            "docs_batch": dict(tokenized_docs),
            "labels": torch.tensor(batch_scores) if len(batch_scores) > 0 else None,
        }
    
class CatDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.q_max_length = 30
        self.d_max_length = 200

    def __call__(self, batch):
        batch_queries = []
        batch_docs = []
        batch_scores = []
        for (q, dx, *args) in batch:
            batch_queries.extend([q]*len(dx))
            batch_docs.extend(dx)
            if len(args) == 0:
                continue
            batch_scores.extend(args[0])
        # flatten lists 
        batch_scores = list(chain.from_iterable(batch_scores))
        batch_queries = list(chain.from_iterable(batch_queries))

        tokenized_sequences = self.tokenizer(
            batch_queries,
            batch_docs,
            padding=True,
            truncation='only_second',
            max_length=self.q_max_length + self.d_max_length,
            return_tensors="pt",
        )
        return {
            "sequences": dict(tokenized_sequences),
            "labels": torch.tensor(batch_scores) if len(batch_scores) > 0 else None,
        }

def _make_pos_pairs(texts):
    output = []
    pos = texts[0]
    for i in range(1, len(texts)):
        output.append([pos, texts[i]])
    return output
    
class DuoDataCollator:
    # creates pairwise input for duoBERT, encoding each possible pair of documents with the query in the form CLS [query] SEP [doc1] SEP [doc2]
    def __init__(self, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __call__(self, batch):
        batch_queries = []
        batch_docs = []
        batch_scores = []
        for (q, dx, *args) in batch:
            batch_queries.append(q)
            batch_document_pairs = _make_pos_pairs(dx)
            batch_docs.append(batch_document_pairs)
            if len(args) == 0:
                continue
            batch_score_pairs = _make_pos_pairs(args[0])
            batch_scores.extend(batch_score_pairs)
            
        # tokenize each pair with each query
        texts = []
        for query, pairs in zip(batch_queries, batch_docs):
            for pair in pairs:
                texts.append(f"[CLS] {query} [SEP] {pair[0]} [SEP] {pair[1]}")
        tokenized_sequences = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            add_special_tokens=True,
        )
                
        return {
            "sequences": dict(tokenized_sequences),
            "labels": torch.tensor(batch_scores).squeeze() if len(batch_scores) > 0 else None,
        }