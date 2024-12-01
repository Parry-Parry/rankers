import os
import shutil
import sys
import logging
from transformers import Trainer
import math
import time
import pandas as pd
from typing import Optional, Union, Dict, List, Callable, Tuple, Any
import datasets
from datasets import Dataset
from transformers import (
    FlaxPreTrainedModel,
    PreTrainedTokenizerBase,
    EvalPrediction,
    TrainerCallback,
    TrainerControl,
    PrinterCallback,
    hf_hub_utils,
)
from transformers.trainer_utils import EvalLoopOutput, speed_metrics, TrainOutput
from transformers.trainer_callback import (
    CallbackHandler,
    DefaultFlowCallback,
    ProgressCallback,
    ExportableState,
)
from transformers.utils import logging, has_length, can_return_loss
from transformers.integrations import get_reporting_integration_callbacks
from functools import partial
import warnings
import jax
from jax import jit
import orbax
import jax.numpy as jnp
import optax
import inspect

from ..loss import LOSS_REGISTRY
from .state import FlaxTrainerState
from ..training_arguments import RankerTrainingArguments

logger = logging.getLogger(__name__)

# heavily inspired by https://github.com/hkjeon13/flax-trainer/blob/main/flax_trainer/trainer.py

LOSS_NAME = "loss.np"
TRAINING_ARGS_NAME = "training_args.pkl"
TRAINER_STATE_NAME = "trainer_state.pkl"
OPTIMIZER_NAME = "optimizer"

DEFAULT_CALLBACKS = [DefaultFlowCallback]
DEFAULT_PROGRESS_CALLBACK = ProgressCallback

'''
TODO:
- Implement call_model_init
- Implement _load_best_model
'''


class FlaxContrastTrainer(Trainer):
    """Customized Trainer from Huggingface's Trainer"""

    def __init__(
        self,
        model: Union[FlaxPreTrainedModel, Any] = None,
        args: RankerTrainingArguments = None,
        loss=None,
        data_collator: Optional["DataCollator"] = None,
        train_dataset: Optional[Union[Dataset, "datasets.Dataset"]] = None,
        eval_dataset: Optional[
            Union[Dataset, Dict[str, Dataset], "datasets.Dataset"]
        ] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], FlaxPreTrainedModel]] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[optax._src.base.Optimizer, optax._src.base.Scheduler] = (
            None,
            None,
        )
    ):
        if args is None:
            output_dir = "tmp_trainer"
            logger.info(
                f"No `TrainingArguments` passed, using `output_dir={output_dir}`."
            )
            args = RankerTrainingArguments(output_dir=output_dir)
        if args.batch_eval_metrics and compute_metrics is not None:
            if (
                "compute_result"
                not in inspect.signature(compute_metrics).parameters.keys()
            ):
                raise ValueError(
                    "When using `batch_eval_metrics`, your `compute_metrics` function must take a `compute_result`"
                    " boolean argument which will be triggered after the last batch of the eval set to signal that the"
                    " summary statistics should be returned by the function."
                )
        self.args = args
        # Seed must be set before instantiating the model when using model
        self.hp_name = None
        self.is_in_train = False

        # set the correct log level depending on the node
        log_level = args.get_process_log_level()
        logging.set_verbosity(log_level)

        if model is None:
            if model_init is not None:
                self.model_init = model_init
                model = self.call_model_init()
            else:
                raise RuntimeError(
                    "`Trainer` requires either a `model` or `model_init` argument"
                )
        else:
            if model_init is not None:
                warnings.warn(
                    "`Trainer` requires either a `model` or `model_init` argument, but not both. `model_init` will"
                    " overwrite your model when calling the `train` method. This will become a fatal error in the next"
                    " release.",
                    FutureWarning,
                )
            self.model_init = model_init

        if getattr(model, "is_parallelizable", False) and getattr(
            model, "model_parallel", False
        ):
            self.is_model_parallel = True
        else:
            self.is_model_parallel = False

        self.data_collator = data_collator
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer

        # Force n_gpu to 1 to avoid DataParallel as MP will manage the GPUs
        if self.is_model_parallel:
            self.args._n_gpu = 1

        # later use `self.model is self.model_wrapped` to check if it's wrapped or not
        self.model_wrapped = model
        self.model = model

        self.optimizer, self.lr_scheduler = optimizers
        if model_init is not None and (
            self.optimizer is not None or self.lr_scheduler is not None
        ):
            raise RuntimeError(
                "Passing a `model_init` is incompatible with providing the `optimizers` argument. "
                "You should subclass `Trainer` and override the `create_optimizer_and_scheduler` method."
            )
        if self.optimizer is None:
            self.create_optimizer_and_scheduler()

        default_callbacks = DEFAULT_CALLBACKS + get_reporting_integration_callbacks(
            self.args.report_to
        )
        callbacks = (
            default_callbacks if callbacks is None else default_callbacks + callbacks
        )
        self.callback_handler = CallbackHandler(
            callbacks, self.model, self.tokenizer, self.optimizer, self.lr_scheduler
        )
        self.add_callback(
            PrinterCallback if self.args.disable_tqdm else DEFAULT_PROGRESS_CALLBACK
        )

        # Will be set to True by `self._setup_loggers()` on first call to `self.log()`.
        self._loggers_initialized = False

        # Create distant repo and output directory if needed
        self.hub_model_id = None
        if self.args.push_to_hub:
            self.init_hf_repo()
        if self.args.should_save:
            os.makedirs(self.args.output_dir, exist_ok=True)

        if not callable(self.data_collator) and callable(
            getattr(self.data_collator, "collate_batch", None)
        ):
            raise ValueError(
                "The `data_collator` should be a simple callable (function, class with `__call__`)."
            )

        if args.max_steps > 0 and args.num_train_epochs > 0:
            logger.warning(
                "max_steps is given, it will override any value given in num_train_epochs"
            )

        if (
            train_dataset is not None
            and not has_length(train_dataset)
            and args.max_steps <= 0
        ):
            raise ValueError(
                "The train_dataset does not implement __len__, max_steps has to be specified. "
                "The number of steps needs to be known in advance for the learning rate scheduler."
            )

        self._signature_columns = None
        self.control = TrainerControl()

        self.state = FlaxTrainerState(
            apply_fn=self.model.__call__,
            params=self.model.params,
            tx=self.optimizer,
            is_local_process_zero=self.is_local_process_zero(),
            is_world_process_zero=self.is_world_process_zero(),
            stateful_callbacks=[
                cb
                for cb in self.callback_handler.callbacks + [self.control]
                if isinstance(cb, ExportableState)
            ],
        )
        # Internal variable to count flos in each process, will be accumulated in `self.state.total_flos` then
        # returned to 0 every time flos need to be logged
        self.current_flos = 0
        self.hp_search_backend = None
        self.can_return_loss = can_return_loss(self.model.__class__)
        self.control = self.callback_handler.on_init_end(
            self.args, self.state, self.control
        )

        # Internal variables to help with automatic batch size reduction
        self._train_batch_size = args.train_batch_size
        self._created_lr_scheduler = False

        if isinstance(loss, str):
            if "flax" not in loss:
                loss = f"flax_{loss}"
            if loss not in LOSS_REGISTRY.available:
                raise ValueError(
                    f"Unknown loss: {loss}, available losses are {LOSS_REGISTRY.available}"
                )
            self.loss = LOSS_REGISTRY.get(loss)
        else:
            self.loss = loss
        self.tokenizer = self.data_collator.tokenizer
        self.model.config.group_size = self.args.group_size

    def create_optimizer_and_scheduler(self, num_training_steps=None):
        self.optimizer = optax.adamw(
            learning_rate=self.args.learning_rate,
            b1=0.9,
            b2=0.98,
            eps=1e-8,
            weight_decay=0.01,
        )
        self.lr_scheduler = (
            optax.cosine_decay_schedule(
                init_value=self.args.learning_rate,
                decay_steps=num_training_steps,
                alpha=0.0,
            )
            if num_training_steps is not None
            else None
        )

    @partial(jit, static_argnums=(0, 2))
    def compute_loss(self, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = self.state.apply_fn(**inputs)
        # Save past state if it exists
        pred = outputs.pred
        if self.args.past_index >= 0:
            self._past = pred[self.args.past_index]

        return (outputs.loss, outputs.pred) if return_outputs else outputs.loss

    def compute_metrics(self, result_frame: pd.DataFrame):
        from ir_measures import evaluator, RR

        qrels = self.eval_dataset.qrels
        metrics = self.args.eval_metrics if self.args.eval_metrics else [RR @ 10]
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

        model = self.model

        batch_size = self.args.eval_batch_size

        logger.info(f"\n***** Running {description} *****")
        logger.info(f"  Num queries = {len(dataset)}")
        logger.info(f"  Batch size = {batch_size}")

        eval_model = model.to_pyterrier()
        result_frame = eval_model.transform(dataset.evaluation_data)
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
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        # handle multipe eval datasets
        override = eval_dataset is not None
        eval_dataset = eval_dataset if override else self.eval_dataset

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

        return output.metrics

    def _load_state(self, checkpoint=None):
        if checkpoint is None:
            return
        state, optimizer, scheduler = orbax.load(checkpoint)
        self.state = state
        self.model.load_state_dict(state.params)
        self.optimizer = optimizer
        self.lr_scheduler = scheduler

    def _save_state(self, checkpoint):
        pass

    def get_num_trainable_parameters(self):
        """
        Get the number of trainable parameters.
        """
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def get_model_param_count(self, trainable_only: bool = False):
        """
        Get the number of parameters of the model.
        """
        if trainable_only:
            return self.get_num_trainable_parameters()
        return sum(p.numel() for p in self.model.parameters())

    def get_learning_rates(self):
        """
        Returns the learning rate of each parameter from self.optimizer.
        """
        if self.optimizer is None:
            raise ValueError(
                "Trainer optimizer is None, please make sure you have setup the optimizer before."
            )
        return [group["lr"] for group in self.optimizer.param_groups]

    def _load_best_model(self):
        logger.info(f"Loading best model from {self.state.best_model_checkpoint} (score: {self.state.best_metric}).")
        best_model_path = os.path.join(self.state.best_model_checkpoint, WEIGHTS_NAME)
        best_safe_model_path = os.path.join(self.state.best_model_checkpoint, SAFE_WEIGHTS_NAME)
        best_adapter_model_path = os.path.join(self.state.best_model_checkpoint, ADAPTER_WEIGHTS_NAME)
        best_safe_adapter_model_path = os.path.join(self.state.best_model_checkpoint, ADAPTER_SAFE_WEIGHTS_NAME)

        model = self.model
        
        if (
            os.path.exists(best_model_path)
            or os.path.exists(best_safe_model_path)
            or os.path.exists(best_adapter_model_path)
            or os.path.exists(best_safe_adapter_model_path)
        ):
            has_been_loaded = True
            weights_only_kwarg = {}
            # If the 'user_content.pt' file does NOT exist, load with the old smp api.
            # Checkpoint must have been saved with the old smp api.
            if self.args.save_safetensors and os.path.isfile(best_safe_model_path):
                state_dict = safetensors.torch.load_file(best_safe_model_path, device="cpu")
            else:
                state_dict = torch.load(
                    best_model_path,
                    map_location="cpu",
                    **weights_only_kwarg,
                )

            load_result = model.load_state_dict(state_dict, strict=True)
            else:
                if _is_peft_model(model):
                    # If train a model using PEFT & LoRA, assume that adapter have been saved properly.
                    # TODO: in the future support only specific min PEFT versions
                    if (hasattr(model, "active_adapter") or hasattr(model, "active_adapters")) and hasattr(
                        model, "load_adapter"
                    ):
                        # For BC for older PEFT versions
                        if hasattr(model, "active_adapters"):
                            active_adapter = model.active_adapters[0]
                            if len(model.active_adapters) > 1:
                                logger.warning("Detected multiple active adapters, will only consider the first one")
                        else:
                            active_adapter = model.active_adapter

                        if os.path.exists(best_adapter_model_path) or os.path.exists(best_safe_adapter_model_path):
                            model.load_adapter(self.state.best_model_checkpoint, active_adapter)
                            # Load_adapter has no return value present, modify it when appropriate.
                            from torch.nn.modules.module import _IncompatibleKeys

                            load_result = _IncompatibleKeys([], [])
                        else:
                            logger.warning(
                                "The intermediate checkpoints of PEFT may not be saved correctly, "
                                f"consider using a custom callback to save {ADAPTER_WEIGHTS_NAME} in corresponding saving folders. "
                                "Check some examples here: https://github.com/huggingface/peft/issues/96"
                            )
                            has_been_loaded = False
                    else:
                        logger.warning("Could not load adapter model, make sure to have `peft>=0.3.0` installed")
                        has_been_loaded = False
                else:
                    # We load the model state dict on the CPU to avoid an OOM error.
                    if os.path.isfile(best_safe_model_path):
                        state_dict = safetensors.torch.load_file(best_safe_model_path, device="cpu")
                    else:
                        state_dict = torch.load(
                            best_model_path,
                            map_location="cpu",
                            **weights_only_kwarg,
                        )
                    load_result = model.load_state_dict(state_dict, False)
                if has_been_loaded:
                    self._issue_warnings_after_load(load_result)
        elif os.path.exists(os.path.join(self.state.best_model_checkpoint, SAFE_WEIGHTS_INDEX_NAME)) or os.path.exists(
            os.path.join(self.state.best_model_checkpoint, WEIGHTS_INDEX_NAME)
        ):
            load_result = load_sharded_checkpoint(
                model, self.state.best_model_checkpoint
            )

    def train(
        self,
        resume_from_checkpoint: Optional[Union[str, bool]] = None,
        trial: Union["optuna.Trial", Dict[str, Any]] = None,
        ignore_keys_for_eval: Optional[List[str]] = None,
        **kwargs,
    ):
        """
        Main training entry point.

        Args:
            resume_from_checkpoint (`str` or `bool`, *optional*):
                If a `str`, local path to a saved checkpoint as saved by a previous instance of [`Trainer`]. If a
                `bool` and equals `True`, load the last checkpoint in *args.output_dir* as saved by a previous instance
                of [`Trainer`]. If present, training will resume from the model/optimizer/scheduler states loaded here.
            trial (`optuna.Trial` or `Dict[str, Any]`, *optional*):
                The trial run or the hyperparameter dictionary for hyperparameter search.
            ignore_keys_for_eval (`List[str]`, *optional*)
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions for evaluation during the training.
            kwargs (`Dict[str, Any]`, *optional*):
                Additional keyword arguments used to hide deprecated arguments
        """
        if resume_from_checkpoint is False:
            resume_from_checkpoint = None

        args = self.args

        self.is_in_train = True

        if "model_path" in kwargs:
            resume_from_checkpoint = kwargs.pop("model_path")
            warnings.warn(
                "`model_path` is deprecated and will be removed in a future version. Use `resume_from_checkpoint` "
                "instead.",
                FutureWarning,
            )
        if len(kwargs) > 0:
            raise TypeError(
                f"train() received got unexpected keyword arguments: {', '.join(list(kwargs.keys()))}."
            )
        # This might change the seed so needs to run first.
        self._train_batch_size = self.args.train_batch_size

        # Model re-init
        model_reloaded = False
        if self.model_init is not None:
            # Seed must be set before instantiating the model when using model_init.
            (
                enable_full_determinism(self.args.seed)
                if self.args.full_determinism
                else seed_everything(self.args.seed)
            )
            self.model = self.call_model_init(trial)
            model_reloaded = True
            # Reinitializes optimizer and scheduler
            self.optimizer, self.lr_scheduler = None, None

        # Load potential model checkpoint
        if isinstance(resume_from_checkpoint, bool) and resume_from_checkpoint:
            resume_from_checkpoint = get_last_checkpoint(args.output_dir)
            if resume_from_checkpoint is None:
                raise ValueError(
                    f"No valid checkpoint found in output directory ({args.output_dir})"
                )

        if resume_from_checkpoint is not None:
            self._load_state(resume_from_checkpoint)
            # In case of repeating the find_executable_batch_size, set `self._train_batch_size` properly
            if self.state.train_batch_size is not None:
                self._train_batch_size = self.state.train_batch_size

        if args.push_to_hub:
            try:
                # Disable progress bars when uploading models during checkpoints to avoid polluting stdout
                hf_hub_utils.disable_progress_bars()
                return self._inner_training_loop(
                    args=args,
                    resume_from_checkpoint=resume_from_checkpoint,
                    trial=trial,
                    ignore_keys_for_eval=ignore_keys_for_eval,
                )
            finally:
                hf_hub_utils.enable_progress_bars()
        else:
            return self._inner_training_loop(
                args=args,
                resume_from_checkpoint=resume_from_checkpoint,
                trial=trial,
                ignore_keys_for_eval=ignore_keys_for_eval,
            )

    def _inner_training_loop(
        self,
        args=None,
        resume_from_checkpoint=None,
        trial=None,
        ignore_keys_for_eval=None,
    ):
        logger.debug(
            f"Currently training with a batch size of: {self._train_batch_size}"
        )
        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()

        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        total_train_batch_size = (
            self._train_batch_size * args.gradient_accumulation_steps * args.world_size
        )

        len_dataloader = None
        num_train_tokens = None
        if has_length(train_dataloader):
            len_dataloader = len(train_dataloader)
            num_update_steps_per_epoch = (
                len_dataloader // args.gradient_accumulation_steps
            )
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            num_examples = self.num_examples(train_dataloader)
            if args.max_steps > 0:
                max_steps = args.max_steps
                num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                    args.max_steps % num_update_steps_per_epoch > 0
                )
                # May be slightly incorrect if the last batch in the training dataloader has a smaller size but it's
                # the best we can do.
                num_train_samples = args.max_steps * total_train_batch_size
                if args.include_tokens_per_second:
                    num_train_tokens = (
                        self.num_tokens(train_dataloader, args.max_steps)
                        * args.gradient_accumulation_steps
                    )
            else:
                max_steps = math.ceil(
                    args.num_train_epochs * num_update_steps_per_epoch
                )
                num_train_epochs = math.ceil(args.num_train_epochs)
                num_train_samples = (
                    self.num_examples(train_dataloader) * args.num_train_epochs
                )
                if args.include_tokens_per_second:
                    num_train_tokens = (
                        self.num_tokens(train_dataloader) * args.num_train_epochs
                    )
        elif (
            args.max_steps > 0
        ):  # Rely on max_steps when dataloader does not have a working size
            max_steps = args.max_steps
            # Setting a very large number of epochs so we go as many times as necessary over the iterator.
            num_train_epochs = sys.maxsize
            num_update_steps_per_epoch = max_steps
            num_examples = total_train_batch_size * args.max_steps
            num_train_samples = args.max_steps * total_train_batch_size
            if args.include_tokens_per_second:
                num_train_tokens = (
                    self.num_tokens(train_dataloader, args.max_steps)
                    * args.gradient_accumulation_steps
                )
        else:
            raise ValueError(
                "args.max_steps must be set to a positive value if dataloader does not have a length, was"
                f" {args.max_steps}"
            )

        # We need to reset the scheduler, as its parameters may be different on subsequent calls
        if self._created_lr_scheduler:
            self.lr_scheduler = None
            self._created_lr_scheduler = False

        self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        # Activate gradient checkpointing if needed
        if args.gradient_checkpointing:
            if args.gradient_checkpointing_kwargs is None:
                gradient_checkpointing_kwargs = {}
            else:
                gradient_checkpointing_kwargs = args.gradient_checkpointing_kwargs

            self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
            )

        # Check if saved optimizer or scheduler states exist
        self._load_state(resume_from_checkpoint)

        self.state = FlaxTrainerState(
            apply_fn=self.model.__call__,
            params=self.model.params,
            tx=self.optimizer,
            is_local_process_zero=self.is_local_process_zero(),
            is_world_process_zero=self.is_world_process_zero(),
            stateful_callbacks=[
                cb
                for cb in self.callback_handler.callbacks + [self.control]
                if isinstance(cb, ExportableState)
            ],
        )
        self.state.train_batch_size = self._train_batch_size

        # Compute absolute values for logging, eval, and save if given as ratio
        if args.logging_steps is not None:
            if args.logging_steps < 1:
                self.state.logging_steps = math.ceil(max_steps * args.logging_steps)
            else:
                self.state.logging_steps = args.logging_steps
        if args.eval_steps is not None:
            if args.eval_steps < 1:
                self.state.eval_steps = math.ceil(max_steps * args.eval_steps)
            else:
                self.state.eval_steps = args.eval_steps
        if args.save_steps is not None:
            if args.save_steps < 1:
                self.state.save_steps = math.ceil(max_steps * args.save_steps)
            else:
                self.state.save_steps = args.save_steps

        model = self.state.params

        # Train!
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples:,}")
        logger.info(f"  Num Epochs = {num_train_epochs:,}")
        logger.info(
            f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size:,}"
        )
        if self.args.per_device_train_batch_size != self._train_batch_size:
            logger.info(
                f"  Training with DataParallel so batch size has been adjusted to: {self._train_batch_size:,}"
            )
        logger.info(
            f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size:,}"
        )
        logger.info(
            f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}"
        )
        logger.info(f"  Total optimization steps = {max_steps:,}")
        logger.info(
            f"  Number of trainable parameters = {self.get_model_param_count(model, trainable_only=True):,}"
        )

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

        # Check if continuing training from a checkpoint
        if resume_from_checkpoint is not None and os.path.isfile(
            os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
        ):
            self.state = FlaxTrainerState.load(
                os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
            )
            self.compare_trainer_and_checkpoint_args(self.args, self.state)
            epochs_trained = int(self.state.global_step // num_update_steps_per_epoch)
            if not args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % (
                    num_update_steps_per_epoch
                )
                steps_trained_in_current_epoch *= args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            logger.info(
                "  Continuing training from checkpoint, will skip to saved global_step"
            )
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(
                f"  Continuing training from global step {self.state.global_step}"
            )
            if not args.ignore_data_skip:
                logger.info(
                    f"  Will skip the first {epochs_trained} epochs then the first"
                    f" {steps_trained_in_current_epoch} batches in the first epoch."
                )

        # Update the references
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader
        # This should be the same if the state has been saved but in case the training arguments changed, it's safer
        # to set this after the load.
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        tr_loss = jnp.array(0.0)
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        grad_norm: Optional[float] = None
        self.control = self.callback_handler.on_train_begin(
            args, self.state, self.control
        )

        if args.eval_on_start:
            self._evaluate(trial, ignore_keys_for_eval, skip_scheduler=True)

        total_batched_samples = 0
        for epoch in range(epochs_trained, num_train_epochs):
            epoch_iterator = train_dataloader
            if hasattr(epoch_iterator, "set_epoch"):
                epoch_iterator.set_epoch(epoch)

            # Reset the past mems state at the beginning of each epoch if necessary.
            if args.past_index >= 0:
                self._past = None

            steps_in_epoch = (
                len(epoch_iterator)
                if len_dataloader is not None
                else args.max_steps * args.gradient_accumulation_steps
            )
            self.control = self.callback_handler.on_epoch_begin(
                args, self.state, self.control
            )

            if (
                epoch == epochs_trained
                and resume_from_checkpoint is not None
                and steps_trained_in_current_epoch == 0
            ):
                self._load_rng_state(resume_from_checkpoint)

            steps_skipped = 0
            if steps_trained_in_current_epoch > 0:
                epoch_iterator = skip_first_batches(
                    epoch_iterator, steps_trained_in_current_epoch
                )
                steps_skipped = steps_trained_in_current_epoch
                steps_trained_in_current_epoch = 0

            step = -1
            for step, inputs in enumerate(epoch_iterator):
                total_batched_samples += 1

                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    if steps_trained_progress_bar is not None:
                        steps_trained_progress_bar.update(1)
                    continue
                elif steps_trained_progress_bar is not None:
                    steps_trained_progress_bar.close()
                    steps_trained_progress_bar = None

                if step % args.gradient_accumulation_steps == 0:
                    self.control = self.callback_handler.on_step_begin(
                        args, self.state, self.control
                    )

                tr_loss_step = self.training_step(inputs)

                if args.logging_nan_inf_filter:
                    # if loss is nan or inf simply add the average of previous logged losses
                    tr_loss = tr_loss.at[0].add(
                        tr_loss
                        / (1 + self.state.global_step - self._globalstep_last_logged)
                    )
                else:

                    tr_loss = tr_loss.at[0].add(float(tr_loss_step))

                self.current_flos += float(self.floating_point_ops(inputs))

                is_last_step_and_steps_less_than_grad_acc = (
                    steps_in_epoch <= args.gradient_accumulation_steps
                    and (step + 1) == steps_in_epoch
                )

                if (
                    total_batched_samples % args.gradient_accumulation_steps == 0
                    or
                    # last step in epoch but step is always smaller than gradient_accumulation_steps
                    is_last_step_and_steps_less_than_grad_acc
                ):

                    # Gradient clipping
                    if args.max_grad_norm is not None and args.max_grad_norm > 0:
                        # TODO: Implement grad norm
                        _grad_norm = None

                    grad_norm = _grad_norm

                    tr_loss_step, grad = jax.value_and_grad(self.loss)(
                        self.state.params
                    )
                    self.state = self.state.apply_gradients(grads=grad)

                    self.control = self.callback_handler.on_optimizer_step(
                        args, self.state, self.control
                    )

                    if self.lr_scheduler is not None:
                        self.lr_scheduler(self.state.step)

                    self.state.global_step += 1
                    self.state.epoch = (
                        epoch + (step + 1 + steps_skipped) / steps_in_epoch
                    )
                    self.control = self.callback_handler.on_step_end(
                        args, self.state, self.control
                    )

                    self._maybe_log_save_evaluate(
                        tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval
                    )
                else:
                    self.control = self.callback_handler.on_substep_end(
                        args, self.state, self.control
                    )

                if self.control.should_epoch_stop or self.control.should_training_stop:
                    break
            if step < 0:
                logger.warning(
                    "There seems to be not a single sample in your epoch_iterator, stopping training at step"
                    f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
                    f" num_steps ({max_steps}) higher than the number of available samples."
                )
                self.control.should_training_stop = True

            self.control = self.callback_handler.on_epoch_end(
                args, self.state, self.control
            )
            self._maybe_log_save_evaluate(
                tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval
            )

            if self.control.should_training_stop:
                break

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info(
            "\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n"
        )
        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            self._load_best_model()

        # add remaining tr_loss
        self._total_loss_scalar += tr_loss
        effective_global_step = max(
            self.state.global_step, 0.001
        )  # Avoid ZeroDivisionError
        train_loss = self._total_loss_scalar / effective_global_step

        metrics = speed_metrics(
            "train",
            start_time,
            num_samples=num_train_samples,
            num_steps=self.state.max_steps,
            num_tokens=num_train_tokens,
        )
        self.store_flos()
        metrics["total_flos"] = self.state.total_flos
        metrics["train_loss"] = train_loss

        self.is_in_train = False

        self.log(metrics)

        run_dir = self._get_output_dir(trial)
        checkpoints_sorted = self._sorted_checkpoints(
            use_mtime=False, output_dir=run_dir
        )

        # Delete the last checkpoint when save_total_limit=1 if it's different from the best checkpoint and process allowed to save.
        if (
            self.args.should_save
            and self.state.best_model_checkpoint is not None
            and self.args.save_total_limit == 1
        ):
            for checkpoint in checkpoints_sorted:
                if not os.path.samefile(checkpoint, self.state.best_model_checkpoint):
                    logger.info(
                        f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit"
                    )
                    shutil.rmtree(checkpoint, ignore_errors=True)

        self.control = self.callback_handler.on_train_end(
            args, self.state, self.control
        )

        # Wait for the checkpoint to be uploaded.
        self._finish_current_push()

        return TrainOutput(self.state.global_step, train_loss, metrics)

    def training_step(self, inputs: Dict[str, Union[jax.Array, Any]]) -> jax.Array:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """

        loss = self.compute_loss(inputs)

        del inputs

        kwargs = {}

        if self.args.n_gpu > 1:
            loss = jnp.mean(loss)  # mean() to average on multi-gpu parallel training

        # TODO: Backward pass NEEDED

        return loss.at[0].divide(self.args.gradient_accumulation_steps)
