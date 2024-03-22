import torch
from torch import nn
import os
import logging
from collections import defaultdict
from transformers import Trainer
from ..modelling.cat import Cat 
from ..modelling.dot import Dot
from .loss import catLoss, dotLoss

logger = logging.getLogger(__name__)

LOSS_NAME = "loss.pt"

class ConstrastTrainer(Trainer):
    """Customized Trainer from Huggingface's Trainer"""

    def __init__(self, *args, loss=None, **kwargs) -> None:
        super(ConstrastTrainer, self).__init__(*args, **kwargs)
        if isinstance(loss, nn.Module): self.loss = loss
        elif isinstance(self.model, Dot): self.loss = dotLoss(loss, kwargs.get("num_negatives", 1), kwargs.get("margin", 1.0))
        elif isinstance(self.model, Cat): self.loss = catLoss(loss, kwargs.get("num_negatives", 1), kwargs.get("margin", 1.0))
        else:
            logger.warning("Model is not Dot or Cat, defaulting to Dot for loss")
            self.loss = dotLoss(loss, kwargs.get("num_negatives", 1), kwargs.get("margin", 1.0))
        self.custom_log = defaultdict(lambda: 0.0)
        self.tokenizer = self.data_collator.tokenizer

    def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch, ignore_keys_for_eval):
        if self.control.should_log:
            log = {}
            for metric in self.customed_log:
                log[metric] = (
                    self._nested_gather(self.custom_log[metric]).mean().item()
                )
                log[metric] = round(
                    (
                        log[metric]
                        / (self.state.global_step - self._globalstep_last_logged)
                        / self.args.gradient_accumulation_steps
                    ),
                    4,
                )
            self.log(log)
            for metric in self.customed_log: self.custom_log[metric] -= self.custom_log[metric]
            self.control.should_log = True
        super()._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)

    def _load_optimizer_and_scheduler(self, checkpoint):
        super()._load_optimizer_and_scheduler(checkpoint)
        if checkpoint is None:
            return
        if os.path.exists(os.path.join(checkpoint, LOSS_NAME)):
            self.loss.load_state_dict(torch.load(os.path.join(checkpoint, LOSS_NAME)))

    def compute_loss(self, model, inputs, return_outputs=False):
        loss_outputs = model(self.loss, **inputs)
        for log_metric in loss_outputs[-1]: self.custom_log[log_metric] += loss_outputs[-1][log_metric]

        components = loss_outputs[:-1]
        loss = components[0] if len(components)==1 else sum(components)
        return loss

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        logger.info("Loading model's weight from %s", resume_from_checkpoint)
        if model: return model.load_state_dict(resume_from_checkpoint)
        else: self.model.load_state_dict(resume_from_checkpoint)