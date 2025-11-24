"""Training infrastructure for neural rankers.

This module provides a customized Trainer class built on HuggingFace's Trainer,
with specialized support for ranking losses, regularization, and IR-specific evaluation.
"""

import logging
import math
import time
from typing import Optional, Union

import pandas as pd
import torch
from datasets import Dataset
from transformers import Trainer
from transformers.integrations.deepspeed import deepspeed_init
from transformers.trainer_utils import EvalLoopOutput, speed_metrics

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
        # Only set compute_metrics if not already provided
        if 'compute_metrics' not in kwargs:
            kwargs['compute_metrics'] = self.compute_metrics
        super().__init__(**kwargs)
        if isinstance(loss_fn, str):
            from .loss import LOSS_REGISTRY

            if loss_fn not in LOSS_REGISTRY.available:
                raise ValueError(
                    f"Unknown loss: {loss_fn}, choices are {LOSS_REGISTRY.available}"
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

                if self.args.regularization not in LOSS_REGISTRY.available:
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

        # Only set tokenizer if data_collator has one
        if hasattr(self.data_collator, 'tokenizer'):
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

    def compute_metrics(self, result_frame: pd.DataFrame, qrels_frame: pd.DataFrame) -> dict:
        """Compute IR metrics for evaluation.

        Uses qrels from the validation dataset to ensure consistency between
        the evaluation data and the qrels used for metric computation.

        Args:
            result_frame: PyTerrier-style result frame with query_id and doc_id

        Returns:
            Dictionary of metric names to values
        """
        from ir_measures import evaluator, nDCG

        result_frame = result_frame.rename(
            columns={"qid": "query_id", "docno": "doc_id"}
        )

        # Use qrels from validation dataset - ensures consistency with result_frame

        metrics = (
            self.args.eval_ir_metrics if self.args.eval_ir_metrics else [nDCG @ 10]
        )
        _evaluator = evaluator(metrics, qrels_frame)
        output = _evaluator.calc_aggregate(result_frame)
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
        model_was_training = model.training
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

        batch_size = self.args.per_device_eval_batch_size

        logger.info(f"\n***** Running {description} *****")
        logger.info(f"  Num queries = {len(dataset)}")
        logger.info(f"  Batch size = {batch_size}")

        try:
            eval_model = model.to_pyterrier(batch_size=batch_size)
            result_frame = eval_model.transform(dataset.data)
            metrics = self.compute_metrics(result_frame, dataset.qrels)

            num_samples = len(dataset)
            metrics = {
                f"{metric_key_prefix}_{k}": v for k, v in metrics.items()
            }

            return EvalLoopOutput(
                predictions=result_frame,
                label_ids=None,
                metrics=metrics,
                num_samples=num_samples,
            )
        finally:
            if model_was_training and hasattr(model, "train"):
                model.train()

    def evaluate(
        self,
        eval_dataset: Optional[Union[Dataset, dict[str, Dataset]]] = None,
        ignore_keys: Optional[list[str]] = None,
        metric_key_prefix: str = "eval",
        **kwargs,  # handle new arguments
    ) -> dict[str, float]:
        """Evaluate the model on the provided dataset.

        Args:
            eval_dataset: Dataset to evaluate on. If None, uses self.eval_dataset.
            ignore_keys: Keys to ignore in the evaluation output.
            metric_key_prefix: Prefix for metric names (default: "eval").
            **kwargs: Additional arguments.

        Returns:
            Dictionary of evaluation metrics.
        """
        if not is_ir_datasets_available():
            raise ImportError(
                "Please install ir_datasets to use the evaluation features."
            )

        # handle multiple eval datasets
        override = eval_dataset is not None
        eval_dataset = eval_dataset if override else self.eval_dataset

        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        start_time = time.time()

        eval_loop = self.evaluation_loop
        output = eval_loop(
            eval_dataset,
            description="Evaluation",
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

    def predict(
        self,
        test_dataset: Dataset,
        ignore_keys: Optional[list[str]] = None,
        metric_key_prefix: str = "test",
    ):
        """Run prediction on test dataset and return predictions and metrics.

        Prediction is identical to evaluation but uses test set prefix instead of
        eval prefix. Like evaluation, this uses the PyTerrier transform interface
        for ranking-specific predictions.

        Args:
            test_dataset: Dataset to run predictions on.
            ignore_keys: Keys to ignore in model output.
            metric_key_prefix: Prefix for metric names (default: "test").

        Returns:
            PredictionOutput with predictions, label_ids, and metrics.
        """
        if not is_ir_datasets_available():
            raise ImportError(
                "Please install ir_datasets to use the prediction features."
            )

        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        start_time = time.time()

        # Temporarily set eval_dataset for compute_metrics if not already set
        original_eval_dataset = self.eval_dataset
        if self.eval_dataset is None:
            self.eval_dataset = test_dataset

        try:
            # Use evaluation_loop which handles the ranking-specific logic
            output = self.evaluation_loop(
                test_dataset,
                description="Prediction",
                prediction_loss_only=None,
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix,
            )
        finally:
            # Restore original eval_dataset
            self.eval_dataset = original_eval_dataset

        total_batch_size = self.args.eval_batch_size * self.args.world_size
        output.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=output.num_samples,
                num_steps=math.ceil(output.num_samples / total_batch_size),
            )
        )

        self.control = self.callback_handler.on_predict(
            self.args, self.state, self.control, output.metrics
        )
        self._memory_tracker.stop_and_update_metrics(output.metrics)

        # Return PredictionOutput compatible with HF Trainer
        from transformers.trainer_utils import PredictionOutput

        return PredictionOutput(
            predictions=output.predictions,
            label_ids=output.label_ids,
            metrics=output.metrics,
        )

    def _maybe_log_save_evaluate(
        self,
        tr_loss,
        grad_norm,
        model,
        trial,
        epoch,
        ignore_keys_for_eval,
        start_time=None,
        learning_rate=None,
    ):
        """
        Evaluate the model and save the best model.

        This method is called during training and properly integrates with HuggingFace's
        best model selection when `use_best_model` is set to True.

        Args:
            tr_loss: Training loss.
            grad_norm: Gradient norm.
            model: Model to evaluate.
            trial: Trial object (for hyperparameter search).
            epoch: Current epoch.
            ignore_keys_for_eval: Keys to ignore during evaluation.
            start_time: Training start time.
            learning_rate: Current learning rate.
        """
        if self.control.should_log and self.state.global_step > self._globalstep_last_logged:
            logs = {}
            tr_loss_scalar = tr_loss.item() if isinstance(tr_loss, torch.Tensor) else tr_loss
            logs["loss"] = round(
                tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged),
                4,
            )
            if grad_norm is not None:
                logs["grad_norm"] = (
                    grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
                )
            if learning_rate is not None:
                logs["learning_rate"] = learning_rate
            else:
                logs["learning_rate"] = self._get_learning_rate()

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.log(logs)

        eval_metrics = None
        if self.control.should_evaluate:
            eval_metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)
            self._report_to_hp_search(trial, self.state.global_step, eval_metrics)

        if self.control.should_save:
            self._save_checkpoint(model, trial)
