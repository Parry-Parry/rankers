import torch
import os
import logging
from transformers import Trainer
import math 
import time
import pandas as pd 
from typing import Optional, Union, Dict, List
from datasets import Dataset
from transformers.trainer_utils import EvalLoopOutput, speed_metrics
from transformers.integrations.deepspeed import deepspeed_init
from .loss import LOSS_REGISTRY

logger = logging.getLogger(__name__)

LOSS_NAME = "loss.pt"

class RankerTrainer(Trainer):
    """Customized Trainer from Huggingface's Trainer"""

    def __init__(self, *args, loss_fn=None, **kwargs) -> None:
        super(RankerTrainer, self).__init__(*args, **kwargs)
        if isinstance(loss_fn, str): 
            if loss_fn not in LOSS_REGISTRY.availible: raise ValueError(f"Unknown loss: {loss_fn}, choices are {LOSS_REGISTRY.availible}")
            self.loss = LOSS_REGISTRY.get(loss_fn)
        else: 
            self.loss = loss_fn
        self.tokenizer = self.data_collator.tokenizer
        self.model.config.group_size = self.args.group_size

    def compute_loss(self, model, inputs, return_outputs=False):
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

        return (loss, outputs) if return_outputs else loss
    
    def compute_metrics(self, result_frame : pd.DataFrame):
        from ir_measures import evaluator, RR
        qrels = self.eval_dataset.qrels
        metrics = self.args.eval_metrics if self.args.eval_metrics else [RR@10]
        evaluator = evaluator(metrics, qrels)

        return evaluator.calc_aggregate(result_frame)

    def evaluation_loop(
        self,
        dataset: Dataset,
        description: str,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Works both with or without labels.
        """
        args = self.args

        # if eval is called w/o train, handle model prep here
        if self.is_deepspeed_enabled and self.deepspeed is None:
            _, _ = deepspeed_init(self, num_training_steps=0, inference=True)

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
        result_frame = eval_model.transform(dataset.evaluation_data)
        metrics = self.compute_metrics(result_frame)

        num_samples = len(dataset)
        metrics = {f"{metric_key_prefix}_{k}": v for k, v in metrics.items()}

        return EvalLoopOutput(predictions=result_frame, label_ids=None, metrics=metrics, num_samples=num_samples)
    
    def evaluate(
        self,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
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

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, output.metrics)

        self._memory_tracker.stop_and_update_metrics(output.metrics)

        return output.metrics

    def _load_optimizer_and_scheduler(self, checkpoint):
        super()._load_optimizer_and_scheduler(checkpoint)
        if checkpoint is None:
            return
        if os.path.exists(os.path.join(checkpoint, LOSS_NAME)):
            self.loss.load_state_dict(torch.load(os.path.join(checkpoint, LOSS_NAME)))

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        logger.info("Loading model's weight from %s", resume_from_checkpoint)
        if model: return model.load_state_dict(resume_from_checkpoint)
        else: self.model.load_state_dict(resume_from_checkpoint)