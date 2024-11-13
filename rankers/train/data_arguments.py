from dataclasses import dataclass, field, fields
from typing import Dict, Any, Optional
import json
from enum import Enum
import torch
from .. import is_ir_datasets_available, is_torch_available, is_pyterrier_available
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)

@dataclass
class RankerDataArguments:
    training_dataset_file : str = field(
        metadata={"help": "Path to the training dataset in a JSONL format (Must be uncompressed)"}
    )
    teacher_file : Optional[str] = field(
        default=None,
        metadata={"help": "Path to the teacher scores in a JSON format"}
    )
    validation_dataset_file : Optional[str] = field(
        default=None,
        metadata={"help": "Path to the validation dataset in a TREC format"}
    )
    test_dataset_file : Optional[str] = field(
        default=None,
        metadata={"help": "Path to the test dataset in a TREC format"}
    )
    ir_dataset : Optional[str] = field(
        default=None,
        metadata={"help": "IR Dataset for text lookup"}
    )
    lazy_load_text : Optional[bool] = field(
        default=False,
        metadata={"help": "Lazy load text from the corpus"}
    )
    precomputed : Optional[bool] = field(
        default=False,
        metadata={"help": "DataFrame with existing text fields"}
    )
    text_field : Optional[str] = field(
        default='text',
        metadata={"help": "Field name for text"}
    )
    query_field : Optional[str] = field(
        default='text',
        metadata={"help": "Field name for query"}
    )
    no_positive : Optional[bool] = field(
        default=False,
        metadata={"help": "Dont use positive samples locatd in 'doc_id_a' column instead use solely 'doc_id_b'"}
    )
    top_k_group : Optional[bool] = field(
        default=False,
        metadata={"help": "Sort by score when doing list-wise ranking and take top-k"}
    )
    eval_ir_dataset : Optional[str] = field(
        default=None,
        metadata={"help": "IR Dataset for evaluation"}
    )

    def __post_init__(self):
        if self.ir_dataset is not None:
            assert is_ir_datasets_available(), "Please install ir_datasets to use the ir_dataset argument"
            try:
                import ir_datasets
                self.ir_dataset = ir_datasets.load(self.ir_dataset)
            except Exception as e:
                raise ValueError(f"Unable to load ir_dataset: {e}")
        if self.eval_ir_dataset is not None:
            assert is_ir_datasets_available(), "Please install ir_datasets to use the eval_ir_dataset argument"
            try:
                import ir_datasets
                self.eval_ir_dataset = ir_datasets.load(self.eval_ir_dataset)
            except Exception as e:
                raise ValueError(f"Unable to load eval_ir_dataset: {e}")
        assert self.training_dataset_file.endswith('jsonl') or self.training_dataset_file.endswith('jsonl.gz'), "Training dataset should be a JSONL file"
        if self.teacher_file:
            assert self.teacher_file.endswith('json') or self.teacher_file.endswith('json.gz'), "Teacher file should be a JSON file"
        if self.validation_dataset_file:
            assert self.validation_dataset_file.endswith(".gz") or self.validation_dataset_file.endswith(".tsv") or self.validation_dataset_file.endswith(".rez"), "Validation dataset should be a TREC formatted run file"
            if is_pyterrier_available():
                import pyterrier as pt
                self.validation_data = pt.io.read_results(self.validation_dataset_file)
            else:
                logging.warning("Pyterrier not available, validation dataset will be loaded as a DataFrame")
                self.validation_data= pd.read_csv(self.validation_dataset_file, sep="\t")
        if self.test_dataset_file:
            assert self.test_dataset_file.endswith(".gz") or self.test_dataset_file.endswith(".tsv") or self.test_dataset_file.endswith(".rez"), "Test dataset should be a TREC formatted run file"
            if is_pyterrier_available():
                import pyterrier as pt
                self.test_data = pt.io.read_results(self.test_dataset_file)
            else:
                logging.warning("Pyterrier not available, test dataset will be loaded as a DataFrame")
                self.test_data = pd.read_csv(self.test_dataset_file, sep="\t")

        
    def to_dict(self):
        """
        Serializes this instance while replace `Enum` by their values (for JSON serialization support). It obfuscates
        the token values by removing their value.
        """
        # filter out fields that are defined as field(init=False)
        d = {field.name: getattr(self, field.name) for field in fields(self) if field.init}

        for k, v in d.items():
            if isinstance(v, Enum):
                d[k] = v.value
            if isinstance(v, list) and len(v) > 0 and isinstance(v[0], Enum):
                d[k] = [x.value for x in v]
            if k.endswith("_token"):
                d[k] = f"<{k.upper()}>"

        return d

    def to_json_string(self):
        """
        Serializes this instance to a JSON string.
        """
        return json.dumps(self.to_dict(), indent=2)

    def to_sanitized_dict(self) -> Dict[str, Any]:
        """
        Sanitized serialization to use with TensorBoard’s hparams
        """
        d = self.to_dict()
        
        valid_types = [bool, int, float, str]
        if is_torch_available():
            valid_types.append(torch.Tensor)

        return {k: v if type(v) in valid_types else str(v) for k, v in d.items()}