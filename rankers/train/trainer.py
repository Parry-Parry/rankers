"""Training infrastructure for neural rankers.

This module provides a customized Trainer class built on HuggingFace's Trainer,
with specialized support for ranking losses, regularization, and IR-specific evaluation.
"""

import torch
import logging
from transformers import Trainer
import math
import time
import pandas as pd
from typing import Optional, Union, Dict, List
from datasets import Dataset
from transformers.trainer_utils import EvalLoopOutput, speed_metrics
from transformers.integrations.deepspeed import deepspeed_init
from .._optional import is_ir_datasets_available

logger = logging.getLogger(__name__)

LOSS_NAME = "loss.pt"


class RankerTrainer(Trainer):
    """Specialized trainer for neural ranking models.

    Extends HuggingFace's Trainer with ranking-specific functionality including:
    - Flexible loss function registration and composition
    - Optional regularization (e.g., FLOPS, sparsity)
    - Custom training loop for bi-encoder models
    - IR evaluation metrics integration

    The trainer handles complex loss configurations, supports both string-based
    loss selection via registry and direct loss function passing, and manages
    compound losses with regularization.

    Args:
        loss_fn (Union[str, callable], optional): Loss function. Can be a string
            key from LOSS_REGISTRY or a callable loss function. Defaults to None.
        **kwargs: Additional arguments passed to Trainer (model, args, datasets, etc.).

    Attributes:
        loss: The main ranking loss function.
        regularize_loss (bool): Whether regularization is enabled.

    Examples:
        Basic training setup::

            from rankers.train import RankerTrainer, RankerTrainingArguments
            from rankers.modelling import Dot
            from rankers.datasets import TrainingDataset

            model = Dot.from_pretrained("bert-base-uncased")
            dataset = TrainingDataset.from_json("train.jsonl")

            args = RankerTrainingArguments(
                output_dir="./output",
                num_train_epochs=3,
                per_device_train_batch_size=32
            )

            trainer = RankerTrainer(
                model=model,
                args=args,
                train_dataset=dataset,
                loss_fn="margeMSE"  # String key from registry
            )

            trainer.train()

        With regularization::

            args = RankerTrainingArguments(
                output_dir="./output",
                regularization="FLOPSLoss",
                q_regularization_weight=0.001,
                regularization_warmup_steps=1000
            )

            trainer = RankerTrainer(
                model=model,
                args=args,
                train_dataset=dataset,
                loss_fn="margeMSE"
            )

        Custom loss function::

            def custom_loss(scores, labels):
                return torch.nn.functional.mse_loss(scores, labels)

            trainer = RankerTrainer(
                model=model,
                args=args,
                train_dataset=dataset,
                loss_fn=custom_loss
            )

    Note:
        The trainer automatically composes regularization losses with the main
        loss when regularization is specified in training arguments.
    """

    def __init__(self, loss_fn=None, **kwargs) -> None:
        super(RankerTrainer, self).__init__(**kwargs)
        if isinstance(loss_fn, str):
            from .loss import LOSS_REGISTRY

            if loss_fn not in LOSS_REGISTRY.availible:
                raise ValueError(
                    f"Unknown loss: {loss_fn}, choices are {LOSS_REGISTRY.availible}"
                )
            self.loss = LOSS_REGISTRY.get(loss_fn)
        else:
            self.loss = loss_fn

        self.regularize_loss = False
        if self.args.regularization is not None:
            # if regularization is callable, use it directly
            from .loss.torch import CompoundLoss

            if callable(self.args.regularization):
                reg_loss = self.args.regularization(
                    self.args.q_regularization_weight,
                    self.args.d_regularization_weight,
                    0,
                    self.args.regularization_warmup_steps,
                )
            else:
                from .loss import LOSS_REGISTRY

                if self.args.regularization not in LOSS_REGISTRY.availible:
                    raise ValueError(
                        f"Unknown regularization: {self.args.regularization}"
                    )
                reg_func = LOSS_REGISTRY.get(self.args.regularization)
                reg_loss = reg_func(
                    self.args.q_regularization_weight,
                    self.args.d_regularization_weight,
                    0,
                    self.args.regularization_warmup_steps,
                )
            self.loss = CompoundLoss([self.loss, reg_loss])
            self.regularize_loss = True

        self.tokenizer = self.data_collator.tokenizer
        self.model.config.group_size = self.args.group_size

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        **kwargs,  # handle new arguments
    ):
        outputs = model(self.loss, **inputs)
        # Save past state if it exists
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if isinstance(outputs, dict) and "loss" not in outputs:
            raise ValueError(
                "The model did not return a loss from the inputs, only the following keys: "
                f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
            )
        # We don't use .loss here since the model may return tuples instead of ModelOutput.
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        to_log = (
            outputs["to_log"]
            if isinstance(outputs, dict) and "to_log" in outputs
            else outputs[1]
        )
        if len(to_log) > 0:
            self.log(to_log)
        return (loss, outputs) if return_outputs else loss

    def compute_metrics(self, result_frame: pd.DataFrame):
        from ir_measures import evaluator, nDCG

        result_frame = result_frame.rename(
            columns={"query_id": "qid", "doc_id": "docno"}
        )

        qrels = pd.DataFrame(self.eval_ir_dataset.qrels_iter())
        metrics = (
            self.args.eval_ir_metrics if self.args.eval_ir_metrics else [nDCG @ 10]
        )
        _evaluator = evaluator(metrics, qrels)
        output = _evaluator.evaluate(result_frame)
        output = {str(k): v for k, v in output.items()}
        return output

    def evaluation_loop(
        self,
        dataset: Dataset,
        description: str,
        metric_key_prefix: str = "val",
        **kwargs,  # handle new arguments
    ) -> EvalLoopOutput:
        """
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Works both with or without labels.
        """
        args = self.args

        # if eval is called w/o train, handle model prep here
        if self.is_deepspeed_enabled and self.deepspeed is None:
            _, _ = deepspeed_init(self, num_training_steps=0, inference=True)
        model = self.model
        if len(self.accelerator._models) == 0 and model is self.model:
            start_time = time.time()
            model = (
                self.accelerator.prepare(model)
                if self.is_deepspeed_enabled
                else self.accelerator.prepare_model(model, evaluation_mode=True)
            )
            self.model_preparation_time = round(time.time() - start_time, 4)

            if self.is_fsdp_enabled:
                self.model = model

            # for the rest of this function `model` is the outside model, whether it was wrapped or not
            if model is not self.model:
                self.model_wrapped = model

            # backward compatibility
            if self.is_deepspeed_enabled:
                self.deepspeed = self.model_wrapped

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        batch_size = self.args.eval_batch_size

        logger.info(f"\n***** Running {description} *****")
        logger.info(f"  Num queries = {len(dataset)}")
        logger.info(f"  Batch size = {batch_size}")

        eval_model = model.to_pyterrier()
        result_frame = eval_model.transform(dataset.data)
        metrics = self.compute_metrics(result_frame)

        num_samples = len(dataset)
        metrics = {f"{metric_key_prefix}_{k}": v for k, v in metrics.items()}

        return EvalLoopOutput(
            predictions=result_frame,
            label_ids=None,
            metrics=metrics,
            num_samples=num_samples,
        )

    def evaluate(
        self,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "test",
        **kwargs,  # handle new arguments
    ) -> Dict[str, float]:
        if not is_ir_datasets_available():
            raise ImportError(
                "Please install ir_datasets to use the evaluation features."
            )

        # handle multipe eval datasets
        override = eval_dataset is not None
        eval_dataset = eval_dataset if override else self.eval_dataset

        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        start_time = time.time()

        eval_loop = self.evaluation_loop
        output = eval_loop(
            eval_dataset,
            description="Evaluation",
            # No point gathering the predictions if there are no metrics, otherwise we defer to
            # self.args.prediction_loss_only
            prediction_loss_only=None,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        total_batch_size = self.args.eval_batch_size * self.args.world_size
        output.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=output.num_samples,
                num_steps=math.ceil(output.num_samples / total_batch_size),
            )
        )

        self.log(output.metrics)
        self.control = self.callback_handler.on_evaluate(
            self.args, self.state, self.control, output.metrics
        )
        self._memory_tracker.stop_and_update_metrics(output.metrics)

        return output.metrics
