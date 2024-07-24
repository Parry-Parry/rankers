from transformers import TrainerCallback
from transformers.integrations import WandbCallback
import ir_measures
import numpy as np
import ir_datasets as irds
import pandas as pd

class EarlyStopping(object):
    def __init__(self, val_topics, metric, qrels, mode='min', min_delta=0, patience=10, percentage=False):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta, percentage)

        if patience == 0:
            self.is_better = lambda a, b: True
            self.step = lambda a: False

        self.val_topics = val_topics
        self.metric = ir_measures.parse_measure(metric)
        self.evaluator = ir_measures.evaluator([self.metric], qrels)

    def step(self, metrics):
        better = False
        if self.best is None:
            self.best = metrics
            return False, True

        if np.isnan(metrics): return True, False

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
            better = True
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True, better
        return False, better

    def _init_is_better(self, mode, min_delta, percentage):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if not percentage:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - min_delta
            if mode == 'max':
                self.is_better = lambda a, best: a > best + min_delta
        else:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - (
                            best * min_delta / 100)
            if mode == 'max':
                self.is_better = lambda a, best: a > best + (
                            best * min_delta / 100)
    
    def compute_metric(self, ranks):
        ranks = ranks.copy().rename(columns={'qid': 'query_id', 'docno': 'doc_id'})
        ranks['score'] = ranks['score'].astype(float)
        ranks['query_id'] = ranks['query_id'].astype(str)
        ranks['doc_id'] = ranks['doc_id'].astype(str)
        value = self.evaluator.calc_aggregate(ranks)
        return list(value.values())[0]
                
    def __call__(self, model):
        ranks = model.transform(self.val_topics)
        value = self.compute_metric(ranks) 
        return *self.step(value), value

class EarlyStoppingCallback(TrainerCallback):

    """
    EarlyStoppingCallback

    Args:
        metric (str): ir_datasets metric
        val_topics (pd.DataFrame): Terrier style topics
        ir_dataset (str): ir_datasets dataset for text lookup
        early_check (int): Check every n steps
        min_train_steps (int): Minimum number of training steps
        mode (str): min or max
        min_delta (int): Minimum change to be considered an improvement
        patience (int): Number of steps to wait before stopping
        percentage (bool): Use percentage change
        log (bool): Log the metric to the state
    """    

    def __init__(self, 
                 metric : str, 
                 ir_dataset : str, 
                 val_topics : pd.DataFrame, 
                 early_check = 10000,
                 every_n_steps = 100000,
                 mode='max', 
                 min_delta=0, 
                 patience=10, 
                 percentage=False,
                 log : bool = False) -> None:
        super().__init__()
        self.metric = metric
        val_topics = val_topics
        corpus = irds.load(ir_dataset)
        queries = pd.DataFrame(corpus.queries_iter()).set_index('query_id').text.to_dict()
        docs = pd.DataFrame(corpus.docs_iter()).set_index('doc_id').text.to_dict()
        qrels = corpus.qrels_iter()
        val_topics['query'] = val_topics['qid'].apply(lambda x: queries[str(x)])
        val_topics['text'] = val_topics['docno'].apply(lambda x: docs[str(x)])
        del queries
        del docs
        self.stopping = EarlyStopping(val_topics, metric, qrels, mode, min_delta, patience, percentage)
        self.early_check = early_check
        self.every_n_steps = every_n_steps
        self.log = log
    
    def on_step_end(self, args, state, control, **kwargs):
        global_step = state.global_step
        if (
            global_step % self.early_check == 0
            and global_step > self.every_n_steps
        ):  
            val_model = kwargs['model'].eval()
            val_model.batch_size = args.per_device_train_batch_size
            stop, better, value = self.stopping(val_model)
            if better: kwargs['model'].save_pretrained(f'{args.output_dir}/best')
            if stop: control.should_training_stop = True  # Stop training
            if self.log: state.log_metrics = {self.metric: value}

class ValidationLoggerCallback(WandbCallback):
    """
    ValidationLoggerCallback

    Args:
        metric (str): ir_datasets metric
        val_topics (pd.DataFrame): Terrier style topics
        ir_dataset (str): ir_datasets dataset for text lookup
    """

    def __init__(self, 
                 metric : str, 
                 val_topics : pd.DataFrame, 
                 ir_dataset : str) -> None:
        super().__init__()
        self.metric = f'val_{metric}'
        val_topics = pd.read_csv(val_topics, sep='\t', index_col=False)
        corpus = irds.load(ir_dataset)
        queries = pd.DataFrame(corpus.queries_iter()).set_index('query_id').text.to_dict()
        docs = pd.DataFrame(corpus.docs_iter()).set_index('doc_id').text.to_dict()
        qrels = corpus.qrels_iter()
        val_topics['query'] = val_topics['qid'].apply(lambda x: queries[str(x)])
        val_topics['text'] = val_topics['docno'].apply(lambda x: docs[str(x)])
        del queries
        del docs
        self.val_topics = val_topics
        self.metric = ir_measures.parse_measure(metric)
        self.evaluator = ir_measures.evaluator([self.metric], qrels)
    
    def compute_metric(self, ranks):
        ranks = ranks.copy().rename(columns={'qid': 'query_id', 'docno': 'doc_id'})
        ranks['score'] = ranks['score'].astype(float)
        ranks['query_id'] = ranks['query_id'].astype(str)
        ranks['doc_id'] = ranks['doc_id'].astype(str)
        value = self.evaluator.calc_aggregate(ranks)
        return list(value.values())[0]

    def on_evaluate(self, args, state, control, **kwargs):
        val_model = kwargs['model'].eval()
        val_model.batch_size = args.per_device_train_batch_size
        ranks = val_model.transform(self.val_topics)
        value = self.compute_metric(ranks)
        self._wandb.log({self.metric: value})