from typing import Any
import numpy as np
from flax.training.common_utils import shard
import jax.numpy as jnp

def process_tokens(tokens):
    return {k : shard(jnp.array(v)) for k, v in tokens.items()}

class FlaxDotDataCollator:
    def __init__(self, 
                 tokenizer, 
                 special_mask=False,
                 q_max_length=30,
                 d_max_length=200,
                 ) -> None:
        self.tokenizer = tokenizer
        self.q_max_length = q_max_length
        self.d_max_length = d_max_length
        self.special_mask = special_mask

    def __call__(self, batch) -> dict:
        batch_queries = []
        batch_docs = []
        batch_scores = []
        for (q, dx, *args) in batch:
            batch_queries.append(q)
            batch_docs.extend(dx)
            if len(args) == 0:
                continue
            batch_scores.extend(args[0])

        tokenized_queries = self.tokenizer(
            batch_queries,
            padding=True,
            truncation=True,
            max_length=self.q_max_length,
            return_tensors="np",
            return_special_tokens_mask=self.special_mask,
        )
        tokenized_docs = self.tokenizer(
            batch_docs,
            padding=True,
            truncation=True,
            max_length=self.d_max_length,
            return_tensors="np",
            return_special_tokens_mask=self.special_mask
        )
 
        return {
            "queries": process_tokens(dict(tokenized_queries)),
            "docs_batch": process_tokens(dict(tokenized_docs)),
            "labels": shard(jnp.array(np.array(batch_scores))) if len(batch_scores) > 0 else None,
        }
    
class FlaxCatDataCollator:
    def __init__(self, 
                 tokenizer,
                 q_max_length=30,
                 d_max_length=200,
                 ) -> None:
        self.tokenizer = tokenizer
        self.q_max_length = q_max_length
        self.d_max_length = d_max_length

    def __call__(self, batch) -> dict:
        batch_queries = []
        batch_docs = []
        batch_scores = []
        for (q, dx, *args) in batch:
            batch_queries.extend([q]*len(dx))
            batch_docs.extend(dx)
            if len(args) == 0:
                continue
            batch_scores.extend(args[0])

        tokenized_sequences = self.tokenizer(
            batch_queries,
            batch_docs,
            padding=True,
            truncation='only_second',
            max_length=self.q_max_length + self.d_max_length,
            return_tensors="np",
        )
        return {
            "sequences": process_tokens(dict(tokenized_sequences)),
            "labels": shard(jnp.array(np.array(batch_scores))) if len(batch_scores) > 0 else None,
        }

def _make_pos_pairs(texts) -> list:
    output = []
    pos = texts[0]
    for i in range(1, len(texts)):
        output.append([pos, texts[i]])
    return output
    
class FlaxPairDataCollator:
    def __init__(self, 
                 tokenizer, 
                 max_length=512) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __call__(self, batch) -> dict:
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
        sequences = [f"[CLS] {query} [SEP] {pair[0]} [SEP] {pair[1]}" for query, pairs in zip(batch_queries, batch_docs) for pair in pairs]

        tokenized_sequences = self.tokenizer(
            sequences,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="np",
            add_special_tokens=True,
        )
                
        return {
            "sequences": process_tokens(dict(tokenized_sequences)),
            "labels": shard(jnp.squeeze(jnp.array(np.array(batch_scores)))) if len(batch_scores) > 0 else None,
        }

class FlaxPromptDataCollator:
    def __init__(self, 
                 tokenizer,
                 prompt : Any,
                 max_length=512,
                 ) -> None:
        self.tokenizer = tokenizer
        self.prompt = prompt
        self.max_length = max_length

    def __call__(self, batch) -> dict:
        batch_queries = []
        batch_docs = []
        batch_scores = []
        for (q, dx, *args) in batch:
            batch_queries.extend([q]*len(dx))
            batch_docs.extend(dx)
            if len(args) == 0:
                continue
            batch_scores.extend(args[0])
        
        sequences = [self.prompt(query=q, doc=d) for q, d in zip(batch_queries, batch_docs)]

        tokenized_sequences = self.tokenizer(
            sequences,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="np",
            add_special_tokens=True,
        )
        return {
            "sequences": process_tokens(dict(tokenized_sequences)),
            "labels": shard(jnp.array(np.array(batch_scores))) if len(batch_scores) > 0 else None,
        }
    
class FlaxPairPromptDataCollator:
    def __init__(self, 
                 tokenizer, 
                 prompt : Any,
                 max_length=512) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.prompt = prompt
    
    def __call__(self, batch) -> dict:
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
        sequences = [self.prompt(query=query, document_1=pair[0], document_2=pair[1]) for query, pairs in zip(batch_queries, batch_docs) for pair in pairs]

        tokenized_sequences = self.tokenizer(
            sequences,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="np",
            add_special_tokens=True,
        )
                
        return {
            "sequences": process_tokens(dict(tokenized_sequences)),
            "labels": shard(jnp.squeeze(jnp.array(np.array(batch_scores)))) if len(batch_scores) > 0 else None,
        }