import re
import base64
import string
import numpy as np
from contextlib import ExitStack
import itertools
from more_itertools import chunked
import torch
import pandas as pd
import pyterrier as pt
from transformers import AutoTokenizer
from pyterrier.model import add_ranks
from ...modelling import Sparse

"""
Taken from https://github.com/thongnt99/learned-sparse-retrieval/blob/main/SparseTransformer/transformer.py
```
"""


class SparseTransformer(pt.Transformer):
    def __init__(
        self,
        model,
        tokenizer,
        device=None,
        batch_size=32,
        text_field="text",
        fp16=False,
        topk=None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model.to(device).eval()
        self.tokenizer = tokenizer
        self.fp16 = fp16
        self.device = device
        all_token_ids = list(range(self.tokenizer.get_vocab_size()))
        self.all_tokens = np.array(self.tokenizer.convert_ids_to_tokens(all_token_ids))
        self.batch_size = batch_size
        self.text_field = text_field
        self.topk = topk

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path,
        device=None,
        batch_size=32,
        text_field="text",
        fp16=False,
        topk=None,
    ):
        model = Sparse.from_pretrained(model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        return cls(model, tokenizer, device, batch_size, text_field, fp16, topk)

    @classmethod
    def from_model(
        cls,
        model,
        tokenizer,
        device=None,
        batch_size=32,
        text_field="text",
        fp16=False,
        topk=None,
    ):
        return cls(model, tokenizer, device, batch_size, text_field, fp16, topk)

    def encode_queries(self, texts, out_fmt="dict", topk=None):
        outputs = []
        if out_fmt != "dict":
            assert topk is None, "topk only supported when out_fmt='dict'"
        with ExitStack() as stack:
            stack.enter_context(torch.no_grad())
            if self.fp16:
                stack.enter_context(torch.cuda.amp.autocast())
            for batch in chunked(texts, self.batch_size):
                enc = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    return_special_tokens_mask=True,
                    return_tensors="pt",
                )
                enc = {k: v.to(self.device) for k, v in enc.items()}
                res = self.model.encode_queries(**enc).cpu().float()
                if out_fmt == "dict":
                    res = self.vec2dicts(res, topk=topk)
                    outputs.extend(res)
                else:
                    outputs.append(res.numpy())
        if out_fmt == "np":
            outputs = np.concatenate(outputs, axis=0)
        elif out_fmt == "np_list":
            outputs = list(itertools.chain.from_iterable(outputs))
        return outputs

    def encode_docs(self, texts, out_fmt="dict", topk=None):
        outputs = []
        if out_fmt != "dict":
            assert topk is None, "topk only supported when out_fmt='dict'"
        with ExitStack() as stack:
            stack.enter_context(torch.no_grad())
            if self.fp16:
                stack.enter_context(torch.cuda.amp.autocast())
            for batch in chunked(texts, self.batch_size):
                enc = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    return_special_tokens_mask=True,
                    return_tensors="pt",
                )
                enc = {k: v.to(self.device) for k, v in enc.items()}
                res = self.model.encode_docs(**enc)
                if out_fmt == "dict":
                    res = self.vec2dicts(res, topk=topk)
                    outputs.extend(res)
                else:
                    outputs.append(res.cpu().float().numpy())
        if out_fmt == "np":
            outputs = np.concatenate(outputs, axis=0)
        elif out_fmt == "np_list":
            outputs = list(itertools.chain.from_iterable(outputs))
        return outputs

    def vec2dicts(self, batch_output, topk=None):
        rtr = []
        idxs, cols = torch.nonzero(batch_output, as_tuple=True)
        weights = batch_output[idxs, cols]
        args = weights.argsort(descending=True)
        idxs = idxs[args]
        cols = cols[args]
        weights = weights[args]
        for i in range(batch_output.shape[0]):
            mask = idxs == i
            col = cols[mask]
            w = weights[mask]
            if topk is not None:
                col = col[:topk]
                w = w[:topk]
            d = {
                self.all_tokens[k]: v
                for k, v in zip(col.cpu().tolist(), w.cpu().tolist())
            }
            rtr.append(d)
        return rtr

    def query_encoder(self, matchop=False, sparse=True, topk=None):
        return SparseQueryEncoder(self, matchop, sparse=sparse, topk=topk or self.topk)

    def doc_encoder(self, text_field=None, sparse=True, topk=None):
        return SparseDocEncoder(
            self, text_field or self.text_field, sparse=sparse, topk=topk or self.topk
        )

    def scorer(self, text_field=None):
        return SparseScorer(self, text_field or self.text_field)

    def transform(self, inp):
        if all(c in inp.columns for c in ["qid", "query", self.text_field]):
            return self.scorer()(inp)
        elif "query" in inp.columns:
            return self.query_encoder()(inp)
        elif self.text_field in inp.columns:
            return self.doc_encoder()(inp)
        raise ValueError(
            f'unsupported columns: {inp.columns}; expecting "query", {repr(self.text_field)}, or both.'
        )


class SparseQueryEncoder(pt.Transformer):
    def __init__(
        self, transformer: SparseTransformer, matchop=False, sparse=True, topk=None
    ):
        self.transformer = transformer
        if not sparse:
            assert not matchop, "matchop only supported when sparse=True"
            assert topk is None, "topk only supported when sparse=True"
        self.matchop = matchop
        self.sparse = sparse
        self.topk = topk

    def encode(self, texts):
        return self.transformer.encode_queries(
            texts, out_fmt="dict" if self.sparse else "np_list", topk=self.topk
        )

    def transform(self, inp):
        res = self.encode(inp["query"])
        if self.matchop:
            res = [_matchop(r) for r in res]
            inp = pt.model.push_queries(inp)
            return inp.assign(query=res)
        if self.sparse:
            return inp.assign(query_toks=res)
        return inp.assign(query_vec=res)


class SparseDocEncoder(pt.Transformer):
    def __init__(
        self, transformer: SparseTransformer, text_field, sparse=True, topk=None
    ):
        self.transformer = transformer
        self.text_field = text_field
        self.sparse = sparse
        if not sparse:
            assert topk is None, "topk only supported when sparse=True"
        self.topk = topk

    def encode(self, texts):
        return self.transformer.encode_docs(
            texts, out_fmt="dict" if self.sparse else "np_list", topk=self.topk
        )

    def transform(self, inp):
        res = self.encode(inp[self.text_field])
        if self.sparse:
            return inp.assign(toks=res)
        return inp.assign(doc_vec=res)


class SparseScorer(pt.Transformer):
    def __init__(self, transformer: SparseTransformer, text_field):
        self.transformer = transformer
        self.text_field = text_field

    def score(self, query_texts, doc_texts):
        q, inv_q = np.unique(
            (
                query_texts.values
                if isinstance(query_texts, pd.Series)
                else np.array(query_texts)
            ),
            return_inverse=True,
        )
        q = self.transformer.encode_queries(q, out_fmt="np")[inv_q]
        d, inv_d = np.unique(
            (
                doc_texts.values
                if isinstance(doc_texts, pd.Series)
                else np.array(doc_texts)
            ),
            return_inverse=True,
        )
        d = self.transformer.encode_docs(d, out_fmt="np")[inv_d]
        return np.einsum("bd,bd->b", q, d)

    def transform(self, inp):
        res = inp.assign(score=self.score(inp["query"], inp[self.text_field]))
        return add_ranks(res)


_alphnum_exp = re.compile(
    "^[" + re.escape(string.ascii_letters + string.digits) + "]+$"
)


def _matchop(d):
    res = []
    for t, w in d.items():
        if not _alphnum_exp.match(t):
            encoded = base64.b64encode(t.encode("utf-8")).decode("utf-8")
            t = f"#base64({encoded})"
        if w != 1:
            t = f"#combine:0={w}({t})"
        res.append(t)
    return " ".join(res)
