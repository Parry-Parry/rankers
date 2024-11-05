from dataclasses import dataclass, field, fields, asdict
from typing import Dict, Any
import json
from enum import Enum
import torch
from .. import is_torch_available


@dataclass
class RankerModelArguments:
    model_name_or_path : str = field(
        metadata={"help": "Huggingface model name or path to model"}
    )

    def __str__(self):
        self_as_dict = asdict(self)

        self_as_dict = {k: f"<{k.upper()}>" if k.endswith("_token") else v for k, v in self_as_dict.items()}

        attrs_as_str = [f"{k}={v},\n" for k, v in sorted(self_as_dict.items())]
        return f"{self.__class__.__name__}(\n{''.join(attrs_as_str)})"

    __repr__ = __str__

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

@dataclass
class RankerDotArguments(RankerModelArguments):
    pooling : str = field(
        default='cls',
        metadata={"help": "Pooling strategy"}
    )
    use_pooler : bool = field(
        default=False,
        metadata={"help": "Whether to use the pooler MLP"}
    )
    model_tied : bool = field(
        default=False,
        metadata={"help": "Whether to tie the weights of the query and document encoder"}
    )
    in_batch_loss : str = field(
        default=None,
        metadata={"help": "Loss function to use for in-batch negatives"}
    )

    def __post_init__(self):
        from .loss import LOSS_REGISTRY
        assert self.pooling in ['cls', 'mean'], "Pooling must be one of 'cls', 'mean'"
        assert self.in_batch_loss is None or self.in_batch_loss in LOSS_REGISTRY.available, f"In-batch loss must be one of {LOSS_REGISTRY.available}"

    

CatArguments = RankerModelArguments