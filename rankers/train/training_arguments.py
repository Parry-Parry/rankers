from transformers import TrainingArguments, AcceleratorConfig
from transformers.utils import is_accelerate_available
from dataclasses import field, fields, dataclass
from enum import Enum
from .. import is_ir_measures_available, is_ir_datasets_available, seed_everything


def parse_ir_measure(measure : str):
    if not is_ir_measures_available(): return measure
    from ir_measures import parse_measure
    return parse_measure(measure)

def get_loss(loss_fn : str):
    from .loss import LOSS_REGISTRY
    if isinstance(loss_fn, str):
        if loss_fn not in LOSS_REGISTRY.availible: raise ValueError(f"Unknown loss: {loss_fn}, choices are {LOSS_REGISTRY.availible}")
        return LOSS_REGISTRY.get(loss_fn)
    else:
        return loss_fn

@dataclass
class RankerTrainingArguments(TrainingArguments):
    group_size : int = field(
        default=2,
        metadata={"help": "Number of documents per query"}
    )
    ir_dataset : str = field(
        default=None,
        metadata={"help": "IR Dataset for text lookup"}
    )

    def __post_init__(self):
        super().__post_init__()
        seed_everything(self.seed)
        assert self.group_size > 0, "Group size must be greater than 0"
        if len(self.eval_metrics) > 0:
            self.eval_metrics = [parse_ir_measure(metric) for metric in self.eval_metrics]
        self.loss_fn = get_loss(self.loss_fn)
        if self.ir_dataset is not None:
            assert is_ir_datasets_available(), "Please install ir_datasets to use the ir_dataset argument"
            try:
                import ir_datasets
                self.ir_dataset = ir_datasets.load(self.ir_dataset)
            except Exception as e:
                raise ValueError(f"Unable to load ir_dataset: {e}")
    
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
            if isinstance(v, list) and is_ir_measures_available():
                import ir_measures
                if type(v[0]) == ir_measures.Measure:
                    d[k] = [x.NAME for x in v]
            if k.endswith("_token"):
                d[k] = f"<{k.upper()}>"
            # Handle the accelerator_config if passed
            if is_accelerate_available() and isinstance(v, AcceleratorConfig):
                d[k] = v.to_dict()
        self._dict_torch_dtype_to_str(d)