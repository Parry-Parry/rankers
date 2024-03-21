from itertools import chain
import random
from torch.utils.data import Dataset
from typing import Any
import json
import pandas as pd
import torch
from typing import Optional, Any
import ir_datasets as irds

from contrast._util import initialise_triples

class TripletDataset(Dataset):
    def __init__(self, 
                 ir_dataset : str,
                 triples : Optional[Any] = None, 
                 teacher_file : Optional[str] = None,
                 num_negatives : int = 0,
                 ) -> None:
        super().__init__()
        self.triples = triples
        self.ir_dataset = irds.load(ir_dataset)
        self.docs = pd.DataFrame(self.ir_dataset.docs_iter()).set_index("doc_id")["text"].to_dict()
        self.queries = pd.DataFrame(self.ir_dataset.queries_iter()).set_index("query_id")["text"].to_dict()

        if self.triples is None:
            self.triples = initialise_triples(self.ir_dataset)
        
        if teacher_file:
            self.teacher = json.load(open(teacher_file, 'r'))

        self.labels = True if teacher_file else False
        self.multi_negatives = True if type(self.triples['doc_id_b'].iloc[0]) == list else False
        if num_negatives > 0 and self.multi_negatives:
            self.triples['doc_id_b'] = self.triples['doc_id_b'].apply(lambda x: random.sample(x, num_negatives))

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        item = self.triples.iloc[idx]
        query = self.queries[item['qid']]
        texts = [self.docs[item['doc_id_a']]]

        if self.multi_negatives:
            texts.extend([self.docs[doc] for doc in item['doc_id_b']])
        else:
            texts.append(self.docs[item['doc_id_b']])

        if self.labels:
            scores = [self.teacher[str(item['qid'])][str(item['doc_id_a'])]]
            if self.multi_negatives:
                scores.extend([self.teacher[str(item['qid'])][str(doc)] for doc in item['doc_id_b']])
            else:
                scores.append(self.teacher[str(item['qid'])][str(item['doc_id_b'])])
            return (query, texts, scores)
        else:
            return (query, texts)

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
        # flatten all lists 
        batch_queries = list(chain.from_iterable(batch_queries))
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
    # duoBERT data collator
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