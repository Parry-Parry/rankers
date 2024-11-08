import random
from torch.utils.data import Dataset
import pandas as pd
import torch
from typing import Optional, Union
import ir_datasets as irds
import json
from .._util import load_json, initialise_irds_eval
from .corpus import Corpus
import gzip

class TrainingDataset(Dataset):
    def __init__(self, 
                 training_dataset_file: str, 
                 corpus: Union[Corpus, irds.Dataset],
                 teacher_data: Optional[dict] = None,
                 group_size: int = 2,
                 no_positive: bool = False
                 ) -> None:
        super().__init__()

        assert training_dataset_file.endswith('jsonl') or training_dataset_file.endswith('jsonl.gz'), \
            "Training dataset should be a JSONL file"

        self.training_dataset_file = training_dataset_file
        self.is_compressed = training_dataset_file.endswith('.gz')
        self.corpus = corpus
        self.teacher_data = teacher_data
        self.group_size = group_size
        self.no_positive = no_positive

        # Collect line offsets if the file is uncompressed
        self.line_offsets = self._get_line_offsets() if not self.is_compressed else None

        self.__post_init__()

    def _get_line_offsets(self):
        """Store byte offsets for each line in an uncompressed JSONL file."""
        offsets = []
        with open(self.training_dataset_file, 'r', encoding="utf-8") as f:
            while True:
                offset = f.tell()
                line = f.readline()
                if not line:
                    break
                offsets.append(offset)
        return offsets

    def _get_line_by_index(self, idx):
        """Retrieve a line by index, using offsets for uncompressed files."""
        if not self.is_compressed:
            with open(self.training_dataset_file, 'r', encoding="utf-8") as f:
                f.seek(self.line_offsets[idx])
                return json.loads(f.readline())
        else:
            # Sequentially read for the compressed case
            data_iterator = self._data_generator()
            for i, line in enumerate(data_iterator):
                if i == idx:
                    return line
            raise IndexError("Index out of range for the dataset.")

    def _data_generator(self):
        """Generator for reading JSON lines from a compressed or uncompressed file."""
        open_fn = gzip.open if self.is_compressed else open
        mode = 'rt' if self.is_compressed else 'r'

        with open_fn(self.training_dataset_file, mode, encoding="utf-8") as f:
            for line in f:
                yield json.loads(line)

    def __post_init__(self):
        assert self.corpus is not None, "Cannot instantiate a text-based dataset without a lookup"
        
        # Initialize documents and queries from corpus
        self.docs = pd.DataFrame(self.corpus.docs_iter()).set_index("doc_id")["text"].to_dict()
        self.queries = pd.DataFrame(self.corpus.queries_iter()).set_index("query_id")["text"].to_dict()

        # Load teacher data if available
        if self.teacher_data:
            self.teacher = load_json(self.teacher_data)
            self.labels = True
        else:
            self.labels = False

        # Use _get_line_by_index to check multi-negative configuration
        first_entry = self._get_line_by_index(0)
        self.multi_negatives = isinstance(first_entry['doc_id_b'], list)

    def __len__(self):
        # Length based on line offsets for uncompressed, or generator count for compressed
        return len(self.line_offsets) if self.line_offsets else sum(1 for _ in self._data_generator())

    def __getitem__(self, idx):
        # Retrieve the line corresponding to idx
        item = self._get_line_by_index(idx)

        # Retrieve query and document texts
        qid, doc_id_a, doc_id_b = item['query_id'], item['doc_id_a'], item['doc_id_b']
        query = self.queries[str(qid)]
        texts = [self.docs[str(doc_id_a)]] if not self.no_positive else []

        # Adjust negatives to fit group_size constraints
        if self.multi_negatives:
            if len(doc_id_b) > (self.group_size - 1):
                doc_id_b = random.sample(doc_id_b, self.group_size - 1)
            texts.extend([self.docs[str(doc)] for doc in doc_id_b])
        else:
            texts.append(self.docs[str(doc_id_b)])

        # Append teacher scores if available
        if self.labels:
            scores = [self._teacher(str(qid), str(doc_id_a), positive=True)] if not self.no_positive else []
            if self.multi_negatives:
                scores.extend([self._teacher(qid, str(doc)) for doc in doc_id_b])
            else:
                scores.append(self._teacher(str(qid), str(doc_id_b)))
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