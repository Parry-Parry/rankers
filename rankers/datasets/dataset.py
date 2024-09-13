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
                 training_data : pd.DataFrame, 
                 corpus : Union[Corpus, irds.Dataset],
                 teacher_file : Optional[str] = None,
                 group_size : int = 2,
                 listwise : bool = False,
                 ) -> None:
        super().__init__()
        self.training_data = training_data
        self.corpus = corpus
        self.teacher_file = teacher_file
        self.group_size = group_size
        self.listwise = listwise

        self.__post_init__()

    
    def __post_init__(self):

        for column in 'query_id', 'doc_id_a', 'doc_id_b':
            if column not in self.training_data.columns: raise ValueError(f"Format not recognised, Column '{column}' not found in triples dataframe")
        self.docs = pd.DataFrame(self.corpus.docs_iter()).set_index("doc_id")["text"].to_dict()
        self.queries = pd.DataFrame(self.corpus.queries_iter()).set_index("query_id")["text"].to_dict()

        if self.teacher_file: self.teacher = load_json(self.teacher_file)

        self.labels = True if self.teacher_file else False
        self.multi_negatives = True if (type(self.training_data['doc_id_b'].iloc[0]) == list) else False

        if not self.listwise:
            if self.group_size > 2 and self.multi_negatives:
                self.training_data['doc_id_b'] = self.training_data['doc_id_b'].map(lambda x: random.sample(x, self.group_size-1))
            elif self.group_size == 2 and self.multi_negatives:
                self.training_data['doc_id_b'] = self.training_data['doc_id_b'].map(lambda x: random.choice(x) if len(x) > 1 else x[0])
                self.multi_negatives = False
            elif self.group_size > 2 and not self.multi_negatives:
                raise ValueError("Group size > 2 not supported for single negative samples")

        self.training_data = [*self.training_data.itertuples(index=False)]

    @classmethod
    def from_irds(cls,
                    ir_dataset : str,
                    teacher_file : Optional[str] = None,
                    group_size : int = 2,
                    collate_fn : Optional[callable] = lambda x : pd.DataFrame(x.docpairs_iter()) 
                    ) -> 'TrainingDataset':
            dataset = irds.load(ir_dataset)
            assert dataset.has_docpairs(), "Dataset does not have docpairs, check you are not using a test collection"
            training_data = collate_fn(dataset)
            return cls(training_data, dataset, teacher_file, group_size)
    
    def __len__(self):
        return len(self.training_data)
    
    def _teacher(self, qid, doc_id, positive=False):
        assert self.labels, "No teacher file provided"
        try: return self.teacher[str(qid)][str(doc_id)] 
        except KeyError: return 0.

    def __getitem__(self, idx):
        item = self.training_data[idx]
        qid, doc_id_a, doc_id_b = item.query_id, item.doc_id_a, item.doc_id_b
        query = self.queries[str(qid)]
        texts = [self.docs[str(doc_id_a)]] if not self.listwise else []

        if self.multi_negatives: texts.extend([self.docs[str(doc)] for doc in doc_id_b])
        else: texts.append(self.docs[str(doc_id_b)])

        if self.labels:
            scores = [self._teacher(str(qid), str(doc_id_a), positive=True)] if not self.listwise else []
            if self.multi_negatives: scores.extend([self._teacher(qid, str(doc)) for doc in doc_id_b])
            else: scores.append(self._teacher(str(qid), str(doc_id_b)))
            return (query, texts, scores)
        else:
            return (query, texts)

class EvaluationDataset(Dataset):
    def __init__(self, 
                 evaluation_data : Union[pd.DataFrame, str], 
                 corpus : Union[Corpus, irds.Dataset]
                 ) -> None:
        super().__init__()
        self.evaluation_data = evaluation_data
        self.corpus = corpus

        self.__post_init__()
    
    def __post_init__(self):
        if type(self.evaluation_data) == str: 
            import pyterrier as pt
            self.evaluation_data = pt.io.read_results(self.evaluation_data)
        else:
            for column in 'qid', 'docno', 'score':
                if column not in self.evaluation_data.columns: raise ValueError(f"Format not recognised, Column '{column}' not found in dataframe")
        self.docs = pd.DataFrame(self.corpus.docs_iter()).set_index("doc_id")["text"].to_dict()
        self.queries = pd.DataFrame(self.corpus.queries_iter()).set_index("query_id")["text"].to_dict()
        self.qrels = pd.DataFrame(self.corpus.qrels_iter())

        self.evaluation_data['text'] = self.evaluation_data['docno'].map(self.docs)
        self.evaluation_data['query'] = self.evaluation_data['qid'].map(self.queries)

    @classmethod
    def from_irds(cls,
                  ir_dataset : str,
                  ) -> 'EvaluationDataset':
            dataset = irds.load(ir_dataset)
            evaluation_data = initialise_irds_eval(dataset)
            return cls(evaluation_data, dataset)
    
    def __len__(self):
        return len(self.evaluation_data.qid.unique())