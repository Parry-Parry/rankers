from collections import defaultdict
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

def _qrel_pivot(frame):
    new = []
    for row in frame.itertuples():
        new.append(
            {
                "qid": row.query_id,
                "docno": row.doc_id,
                "score": row.relevance
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
        lookup = defaultdict(dict)
        for qid, group in scores.groupby('qid'):
            for docno, score in zip(group['docno'], group['score']):
                lookup[qid][docno] = score
        return lookup

def initialise_irds_eval(dataset : irds.Dataset):
    qrels = pd.DataFrame(dataset.qrels_iter())
    return _qrel_pivot(qrels)

def load_json(file: str):
    import json
    import gzip
    """
    Load a JSON or JSONL (optionally compressed with gzip) file.

    Parameters:
    file (str): The path to the file to load.

    Returns:
    dict or list: The loaded JSON content. Returns a list for JSONL files, 
                  and a dict for JSON files.

    Raises:
    ValueError: If the file extension is not recognized.
    """
    if file.endswith(".json"):
        with open(file, 'r') as f:
            return json.load(f)
    elif file.endswith(".jsonl"):
        with open(file, 'r') as f:
            return [json.loads(line) for line in f]
    elif file.endswith(".json.gz"):
        with gzip.open(file, 'rt') as f:
            return json.load(f)
    elif file.endswith(".jsonl.gz"):
        with gzip.open(file, 'rt') as f:
            return [json.loads(line) for line in f]
    else:
        raise ValueError(f"Unknown file type for {file}")

def save_json(data, file: str):
    import json
    import gzip
    """
    Save data to a JSON or JSONL file (optionally compressed with gzip).

    Parameters:
    data (dict or list): The data to save. Must be a list for JSONL files.
    file (str): The path to the file to save.

    Raises:
    ValueError: If the file extension is not recognized.
    """
    if file.endswith(".json"):
        with open(file, 'w') as f:
            json.dump(data, f)
    elif file.endswith(".jsonl"):
        with open(file, 'w') as f:
            for item in data:
                f.write(json.dumps(item) + '\n')
    elif file.endswith(".json.gz"):
        with gzip.open(file, 'wt') as f:
            json.dump(data, f)
    elif file.endswith(".jsonl.gz"):
        with gzip.open(file, 'wt') as f:
            for item in data:
                f.write(json.dumps(item) + '\n')
    else:
        raise ValueError(f"Unknown file type for {file}")
        
    