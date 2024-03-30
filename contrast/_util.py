import logging
from typing import Optional
import pandas as pd
import pyterrier as pt
import ir_datasets as irds

logger = logging.getLogger(__name__)

def _pivot(frame, negatives = None):
    new = []
    for row in frame.itertuples():
        new.append(
            {
                "qid": row.query_id,
                "docno": row.doc_id_a,
                "pos": 1
            })
        if negatives:
            for doc in negatives[row.query_id]:
                new.append(
                    {
                        "qid": row.query_id,
                        "docno": doc
                    })
        else:
            new.append(
                {
                    "qid": row.query_id,
                    "docno": row.doc_id_b
                })
    return pd.DataFrame.from_records(new)

def get_teacher_scores(model : pt.Transformer, 
                       corpus : Optional[pd.DataFrame] = None, 
                       ir_dataset : Optional[str] = None, 
                       subset : Optional[int] = None, 
                       negatives : Optional[dict] = None,
                       seed : int = 42):
        assert corpus is not None or ir_dataset is not None, "Either corpus or ir_dataset must be provided"
        if corpus:
            for column in ["query", "text"]: assert column in corpus.columns, f"{column} not found in corpus"
        if ir_dataset:
            dataset = irds.load(ir_dataset)
            docs = pd.DataFrame(dataset.docs_iter()).set_index("doc_id")["text"].to_dict()
            queries = pd.DataFrame(dataset.queries_iter()).set_index("query_id")["text"].to_dict()
            corpus = pd.DataFrame(dataset.docpairs_iter())
            if negatives:
                corpus = corpus[['query_id', 'doc_id_a']]
            corpus = _pivot(corpus, negatives)
            corpus['text'] = corpus['docno'].map(docs)
            corpus['query'] = corpus['qid'].map(queries)
            if subset:
                corpus = corpus.sample(n=subset, random_state=seed)

        logger.warning("Retrieving scores, this may take a while...")
        scores = model.transform(corpus)
        return scores

def initialise_triples(dataset : irds.Dataset):
    triples = pd.DataFrame(dataset.docpairs_iter())
    return _pivot(triples)