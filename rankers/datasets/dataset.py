import random
from torch.utils.data import Dataset
import pandas as pd
import torch
from typing import Optional, Union
import ir_datasets as irds
from .._util import load_json, initialise_irds_eval
from .corpus import Corpus

class TrainingDataset(Dataset):
    def __init__(self, 
                 training_dataset : pd.DataFrame, 
                 corpus : Union[Corpus, irds.Dataset],
                 teacher_data : Optional[dict] = None,
                 group_size : int = 2,
                 use_positive : bool = False,
                 ) -> None:
        super().__init__()
        self.training_dataset = training_dataset
        self.corpus = corpus
        self.teacher_data = teacher_data
        self.group_size = group_size
        self.use_positive = use_positive

        self.__post_init__()

    def __post_init__(self):
        assert self.corpus is not None, "Cannot instantiate a text-based dataset without a lookup"
        for column in 'query_id', 'doc_id_a', 'doc_id_b':
            if column not in self.training_dataset.columns: raise ValueError(f"Format not recognised, Column '{column}' not found in triples dataframe")
        self.docs = pd.DataFrame(self.corpus.docs_iter()).set_index("doc_id")["text"].to_dict()
        self.queries = pd.DataFrame(self.corpus.queries_iter()).set_index("query_id")["text"].to_dict()

        if self.teacher_data: self.teacher = load_json(self.teacher_data)

        self.labels = True if self.teacher_data else False
        self.multi_negatives = True if (type(self.training_dataset['doc_id_b'].iloc[0]) == list) else False

        if self.use_positive:
            if self.group_size > 2 and self.multi_negatives:
                self.training_dataset['doc_id_b'] = self.training_dataset['doc_id_b'].map(lambda x: random.sample(x, self.group_size-1))
            elif self.group_size == 2 and self.multi_negatives:
                self.training_dataset['doc_id_b'] = self.training_dataset['doc_id_b'].map(lambda x: random.choice(x) if len(x) > 1 else x[0])
                self.multi_negatives = False
            elif self.group_size > 2 and not self.multi_negatives:
                raise ValueError("Group size > 2 not supported for single negative samples")

        self.training_dataset = [*self.training_dataset.itertuples(index=False)]

    @classmethod
    def from_irds(cls,
                    ir_dataset : irds.Dataset,
                    teacher_data : Optional[dict] = None,
                    group_size : int = 2,
                    collate_fn : Optional[callable] = lambda x : pd.DataFrame(x.docpairs_iter()) 
                    ) -> 'TrainingDataset':
            assert ir_dataset.has_docpairs(), "Dataset does not have docpairs, check you are not using a test collection"
            training_dataset = collate_fn(ir_dataset)
            return cls(training_dataset, ir_dataset, teacher_data, group_size)
    
    def __len__(self):
        return len(self.training_dataset)
    
    def _teacher(self, qid, doc_id, positive=False):
        assert self.labels, "No teacher file provided"
        try: return self.teacher[str(qid)][str(doc_id)] 
        except KeyError: return 0.

    def __getitem__(self, idx):
        item = self.training_dataset[idx]
        qid, doc_id_a, doc_id_b = item.query_id, item.doc_id_a, item.doc_id_b
        query = self.queries[str(qid)]
        texts = [self.docs[str(doc_id_a)]] if self.use_positive else []

        if self.multi_negatives: texts.extend([self.docs[str(doc)] for doc in doc_id_b])
        else: texts.append(self.docs[str(doc_id_b)])

        if self.labels:
            scores = [self._teacher(str(qid), str(doc_id_a), positive=True)]  if self.use_positive else []
            if self.multi_negatives: scores.extend([self._teacher(qid, str(doc)) for doc in doc_id_b])
            else: scores.append(self._teacher(str(qid), str(doc_id_b)))
            return (query, texts, scores)
        else:
            return (query, texts)

class EvaluationDataset(Dataset):
    def __init__(self, 
                 evaluation_dataset : pd.DataFrame, 
                 corpus : Union[Corpus, irds.Dataset]
                 ) -> None:
        super().__init__()
        self.evaluation_dataset = evaluation_dataset
        self.corpus = corpus

        self.__post_init__()
    
    def __post_init__(self):


        for column in 'qid', 'docno', 'score':
            if column not in self.evaluation_dataset.columns: raise ValueError(f"Format not recognised, Column '{column}' not found in dataframe")
        self.docs = pd.DataFrame(self.corpus.docs_iter()).set_index("doc_id")["text"].to_dict()
        self.queries = pd.DataFrame(self.corpus.queries_iter()).set_index("query_id")["text"].to_dict()
        self.qrels = pd.DataFrame(self.corpus.qrels_iter())

        self.evaluation_dataset['text'] = self.evaluation_dataset['docno'].map(self.docs)
        self.evaluation_dataset['query'] = self.evaluation_dataset['qid'].map(self.queries)

    @classmethod
    def from_irds(cls,
                  ir_dataset : irds.Dataset,
                  ) -> 'EvaluationDataset':
            evaluation_dataset = initialise_irds_eval(ir_dataset)
            return cls(evaluation_dataset, ir_dataset)
    
    def __len__(self):
        return len(self.evaluation_dataset.qid.unique())