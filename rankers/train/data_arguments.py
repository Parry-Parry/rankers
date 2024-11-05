from dataclasses import dataclass, field, fields
from typing import Dict, Any
import json
from enum import Enum
import torch
from .. import is_ir_datasets_available, is_torch_available, is_pyterrier_available
from .._util import load_json
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)

@dataclass
class RankerDataArguments:
    training_dataset_file : str = field(
        metadata={"help": "Path to the training dataset"}
    )
    teacher_file : str = field(
        default=None,
        metadata={"help": "Path to the teacher scores"}
    )
    validation_dataset_file : str = field(
        default=None,
        metadata={"help": "Path to the validation dataset"}
    )
    test_dataset_file : str = field(
        default=None,
        metadata={"help": "Path to the test dataset"}
    )
    ir_dataset : str = field(
        default=None,
        metadata={"help": "IR Dataset for text lookup"}
    )
    use_positive : bool = field(
        default=False,
        metadata={"help": "Use positive samples locatd in 'doc_id_a' column otherwise use solely 'doc_id_b'"}
    )

    def __post_init__(self):
        if self.ir_dataset is not None:
            assert is_ir_datasets_available(), "Please install ir_datasets to use the ir_dataset argument"
            try:
                import ir_datasets
                self.ir_dataset = ir_datasets.load(self.ir_dataset)
            except Exception as e:
                raise ValueError(f"Unable to load ir_dataset: {e}")
        assert self.training_dataset_file.endswith('jsonl') or self.training_dataset_file.endswith('jsonl.gz'), "Training dataset should be a JSONL file"
        self.training_dataset = load_json(self.training_dataset_file)
        if self.teacher_file:
            assert self.teacher_file.endswith('json') or self.teacher_file.endswith('json.gz'), "Teacher file should be a JSON file"
            self.teacher_data = load_json(self.teacher_file)
        if self.validation_dataset_file:
            assert self.validation_dataset_file.endswith(".gz") or self.validation_dataset_file.endswith(".tsv") or self.validation_dataset_file.endswith(".rez"), "Validation dataset should be a TREC formatted run file"
            if is_pyterrier_available():
                import pyterrier as pt
                self.validation_dataset = pt.io.read_results(self.validation_dataset_file)
            else:
                logging.warning("Pyterrier not available, validation dataset will be loaded as a DataFrame")
                self.validation_dataset = pd.read_csv(self.validation_dataset_file, sep="\t")
        if self.test_dataset_file:
            assert self.test_dataset_file.endswith(".gz") or self.test_dataset_file.endswith(".tsv") or self.test_dataset_file.endswith(".rez"), "Test dataset should be a TREC formatted run file"
            if is_pyterrier_available():
                import pyterrier as pt
                self.test_dataset = pt.io.read_results(self.test_dataset_file)
            else:
                logging.warning("Pyterrier not available, test dataset will be loaded as a DataFrame")
                self.test_dataset = pd.read_csv(self.test_dataset_file, sep="\t")

        
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
        d = {**d, **{"train_batch_size": self.train_batch_size, "eval_batch_size": self.eval_batch_size}}

        valid_types = [bool, int, float, str]
        if is_torch_available():
            valid_types.append(torch.Tensor)

        return {k: v if type(v) in valid_types else str(v) for k, v in d.items()}