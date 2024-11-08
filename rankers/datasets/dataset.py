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
                 training_dataset : pd.DataFrame, 
                 corpus : Union[Corpus, irds.Dataset],
                 teacher_data : Optional[dict] = None,
                 group_size : int = 2,
                 no_positive : bool = False,
                 shuffle_buffer_size : int = 10000
                 ) -> None:
        super().__init__()
        self.training_dataset = training_dataset
        self.corpus = corpus
        self.teacher_data = teacher_data
        self.group_size = group_size
        self.no_positive = no_positive
        self.shuffle_buffer_size = shuffle_buffer_size

        self.__post_init__()

    def _data_generator(self):
        # Open the file in the appropriate mode based on its extension
        if self.training_dataset_file.endswith('.gz'):
            open_fn = gzip.open
            mode = 'rt'
        else:
            open_fn = open
            mode = 'r'
        
        with open_fn(self.training_dataset_file, mode, encoding="utf-8") as f:
            for line in f:
                # Parse each line into a JSON object (dictionary)
                yield json.loads(line)

    This error occurs because first_entry is being accessed as if it were a dictionary, but it seems to be a string instead. This usually happens when an object is read in line-by-line but isn't properly parsed as JSON. Since first_entry should be a parsed JSON object (a dictionary), we need to ensure each line is read and interpreted correctly.

To fix this, let's make sure each line is parsed into JSON in _data_generator. We should also ensure that first_entry is correctly set up as a parsed dictionary in __post_init__.

Here’s how to modify your _data_generator to handle this:

def _data_generator(self):
    # Open the file in the appropriate mode based on its extension
    if self.training_dataset_file.endswith('.gz'):
        open_fn = gzip.open
        mode = 'rt'
    else:
        open_fn = open
        mode = 'r'
    
    with open_fn(self.training_dataset_file, mode, encoding="utf-8") as f:
        for line in f:
            # Parse each line into a JSON object (dictionary)
            yield json.loads(line)

    def __post_init__(self):
        assert self.corpus is not None, "Cannot instantiate a text-based dataset without a lookup"

        # Load corpus documents and queries
        self.docs = pd.DataFrame(self.corpus.docs_iter()).set_index("doc_id")["text"].to_dict()
        self.queries = pd.DataFrame(self.corpus.queries_iter()).set_index("query_id")["text"].to_dict()

        # Load teacher data if available
        if self.teacher_data:
            self.teacher = load_json(self.teacher_data)
            self.labels = True
        else:
            self.labels = False

        # Initialize a generator and get the first entry to check if doc_id_b is a list
        data_iterator = self._data_generator()
        first_entry = next(data_iterator, None)
        
        # Ensure first_entry is a dictionary and check for multi-negative setup
        if first_entry:
            self.multi_negatives = isinstance(first_entry['doc_id_b'], list)
        else:
            self.multi_negatives = False

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
        # Streaming makes it tricky to get the length; it could be implemented by counting lines
        with open(self.training_dataset_file, 'r') as f:
            return sum(1 for _ in f)
    
    def _teacher(self, qid, doc_id, positive=False):
        assert self.labels, "No teacher file provided"
        try: return self.teacher[str(qid)][str(doc_id)] 
        except KeyError: return 0.

    def __getitem__(self, idx):
        # Populate the buffer for approximate shuffling, as previously defined
        buffer = []
        data_iterator = self._data_generator()  # Reinitialize to reset stream for each access

        try:
            for _ in range(self.shuffle_buffer_size):
                buffer.append(next(data_iterator))
            random.shuffle(buffer)
        except StopIteration:
            random.shuffle(buffer)

        item = buffer[idx % len(buffer)]

        # Retrieve IDs and query text
        qid, doc_id_a, doc_id_b = item['query_id'], item['doc_id_a'], item['doc_id_b']
        query = self.queries[str(qid)]
        texts = [self.docs[str(doc_id_a)]] if not self.no_positive else []

        # Handle multi-negative or single-negative cases
        if self.multi_negatives:
            # Adjust the number of negatives to match group_size - 1
            if len(doc_id_b) > (self.group_size - 1):
                doc_id_b = random.sample(doc_id_b, self.group_size - 1)
            texts.extend([self.docs[str(doc)] for doc in doc_id_b])
        else:
            texts.append(self.docs[str(doc_id_b)])

        # Add teacher scores if labels are available
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