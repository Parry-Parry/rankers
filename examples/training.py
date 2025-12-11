"""
Training Examples for Rankers

This module demonstrates various training configurations and techniques using the RankerTrainer.
Examples cover different loss functions, data loading patterns, and trainer configurations.
"""

import torch
from transformers import HfArgumentParser

from rankers import (
    Cat,
    CatDataCollator,
    Dot,
    DotConfig,
    DotDataCollator,
    RankerDataArguments,
    RankerDotArguments,
    RankerModelArguments,
    RankerTrainer,
    RankerTrainingArguments,
    TrainingDataset,
)


# Example 1: Basic Dot-product model training with margin MSE loss
def train_dot_basic_example():
    """
    Basic training example using Dot-product model with margin MSE loss.

    This demonstrates:
    - Loading a pretrained model with custom config
    - Creating a training dataset
    - Setting up trainer with basic configuration
    - Training and saving the model
    """
    # Parse arguments from command line
    parser = HfArgumentParser((RankerDotArguments, RankerDataArguments, RankerTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Create model config with custom settings
    model_config = DotConfig.from_pretrained(
        model_args.model_name_or_path,
        pooling_type=model_args.pooling_type,
        model_tied=model_args.model_tied,
        use_pooler=model_args.use_pooler,
        inbatch_loss=model_args.inbatch_loss,
    )
    model = Dot.from_pretrained(model_args.model_name_or_path, config=model_config)

    # Load training dataset
    # group_size is inferred from the dataset's first entry
    dataset = TrainingDataset(
        data_args.training_dataset_file,
        corpus=data_args.ir_dataset,
        no_positive=data_args.no_positive,
        teacher_file=data_args.teacher_file,
    )

    # Set up data collator
    collate_fn = DotDataCollator(model.tokenizer)

    # Create trainer with loss function specified
    trainer = RankerTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collate_fn,
        loss_fn=training_args.loss_fn,  # Loss specified from args
    )

    # Train the model
    trainer.train()

    # Save trained model
    trainer.save_model(training_args.output_dir)


# Example 2: Training with custom loss function
def train_with_custom_loss_example():
    """
    Training with a specific loss function specified programmatically.

    This demonstrates:
    - Explicitly setting a loss function
    - Training with margin MSE loss
    - Accessing loss registry
    """
    parser = HfArgumentParser((RankerDotArguments, RankerDataArguments, RankerTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Override loss function
    training_args.loss_fn = "margin_mse"

    model_config = DotConfig.from_pretrained(
        model_args.model_name_or_path,
        pooling_type=model_args.pooling_type,
    )
    model = Dot.from_pretrained(model_args.model_name_or_path, config=model_config)

    # group_size is inferred from the dataset's first entry
    dataset = TrainingDataset(
        data_args.training_dataset_file,
    )

    collate_fn = DotDataCollator(model.tokenizer)

    trainer = RankerTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collate_fn,
        loss_fn="margin_mse",
    )

    trainer.train()
    trainer.save_model(training_args.output_dir)


# Example 3: Cat model training with different configuration
def train_cat_example():
    """
    Training a concatenation-based (Cat) ranker model.

    This demonstrates:
    - Loading a Cat model (concatenates query and document)
    - Using Cat-specific data collator
    - Training with default loss function
    """
    parser = HfArgumentParser((RankerModelArguments, RankerDataArguments, RankerTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Load Cat model
    model = Cat.from_pretrained(model_args.model_name_or_path)

    # group_size is inferred from the dataset's first entry
    dataset = TrainingDataset(
        data_args.training_dataset_file,
        corpus=data_args.ir_dataset,
    )

    # Use Cat-specific data collator
    collate_fn = CatDataCollator(model.tokenizer)

    trainer = RankerTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collate_fn,
        loss_fn=training_args.loss_fn,
    )

    trainer.train()
    trainer.save_model(training_args.output_dir)


# Example 4: Training with gradient accumulation and mixed precision
def train_with_optimization_example():
    """
    Training with optimization techniques for large models.

    This demonstrates:
    - Gradient accumulation for larger effective batch sizes
    - Mixed precision training for memory efficiency
    - Custom learning rate and warmup settings
    """
    parser = HfArgumentParser((RankerDotArguments, RankerDataArguments, RankerTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Set optimization parameters
    training_args.gradient_accumulation_steps = 4
    training_args.fp16 = torch.cuda.is_available()
    training_args.learning_rate = 1e-5
    training_args.warmup_steps = 1000
    training_args.weight_decay = 0.01

    model_config = DotConfig.from_pretrained(model_args.model_name_or_path)
    model = Dot.from_pretrained(model_args.model_name_or_path, config=model_config)

    dataset = TrainingDataset(data_args.training_dataset_file)

    collate_fn = DotDataCollator(model.tokenizer)

    trainer = RankerTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collate_fn,
        loss_fn="margin_mse",
    )

    trainer.train()
    trainer.save_model(training_args.output_dir)


# Example 5: Training with custom group size and regularization
def train_with_regularization_example():
    """
    Training with custom group size and regularization loss.

    This demonstrates:
    - Explicit group size for training (overriding dataset inference)
    - Adding regularization loss (e.g., FLOPs regularization)
    - Fine-tuning pre-trained models
    """
    parser = HfArgumentParser((RankerDotArguments, RankerDataArguments, RankerTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Set regularization
    training_args.regularization = "flops_reg"
    training_args.regularization_weight = 0.01

    model_config = DotConfig.from_pretrained(
        model_args.model_name_or_path,
        model_tied=True,
    )
    model = Dot.from_pretrained(model_args.model_name_or_path, config=model_config)

    # Specify group_size=8 to use 1 positive + 7 negatives per query
    dataset = TrainingDataset(
        data_args.training_dataset_file,
        group_size=8,
    )

    collate_fn = DotDataCollator(model.tokenizer)

    trainer = RankerTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collate_fn,
        loss_fn="margin_mse",
    )

    trainer.train()
    trainer.save_model(training_args.output_dir)


# Example 6: Training with checkpoint management and best model selection
def train_with_checkpointing_example():
    """
    Training with checkpoint management and automatic best model selection.

    This demonstrates:
    - Saving checkpoints at regular intervals
    - Loading the best checkpoint based on evaluation metrics
    - Custom save and eval strategy
    """
    parser = HfArgumentParser((RankerDotArguments, RankerDataArguments, RankerTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Configure checkpointing
    training_args.save_strategy = "steps"
    training_args.save_steps = 500
    training_args.eval_strategy = "steps"
    training_args.eval_steps = 500
    training_args.load_best_model_at_end = True
    training_args.metric_for_best_model = "loss"
    training_args.save_total_limit = 3

    model_config = DotConfig.from_pretrained(model_args.model_name_or_path)
    model = Dot.from_pretrained(model_args.model_name_or_path, config=model_config)

    dataset = TrainingDataset(data_args.training_dataset_file)

    collate_fn = DotDataCollator(model.tokenizer)

    trainer = RankerTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collate_fn,
        loss_fn="margin_mse",
    )

    trainer.train()
    trainer.save_model(training_args.output_dir)


# Example 7: Multi-GPU training with distributed setup
def train_distributed_example():
    """
    Training with distributed setup across multiple GPUs.

    This demonstrates:
    - Multi-GPU training configuration
    - Distributed data parallel setup
    - Using torch.distributed launcher
    """
    parser = HfArgumentParser((RankerDotArguments, RankerDataArguments, RankerTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Set distributed training parameters
    training_args.ddp_backend = "nccl"
    training_args.local_rank = -1  # Will be set by launcher

    model_config = DotConfig.from_pretrained(model_args.model_name_or_path)
    model = Dot.from_pretrained(model_args.model_name_or_path, config=model_config)

    dataset = TrainingDataset(data_args.training_dataset_file)

    collate_fn = DotDataCollator(model.tokenizer)

    trainer = RankerTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collate_fn,
        loss_fn="margin_mse",
    )

    trainer.train()
    trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    # Run basic training example
    train_dot_basic_example()
