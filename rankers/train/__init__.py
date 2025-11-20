"""Training framework for neural ranking models.

This module provides a complete training infrastructure for neural rankers built on top of
HuggingFace's Trainer. It includes customized trainers, training arguments, and a flexible
loss function framework.

Key Components:
    - **trainer**: RankerTrainer with specialized training logic
    - **training_arguments**: RankerTrainingArguments for training configuration
    - **model_arguments**: Model-specific arguments for different architectures
    - **data_arguments**: Dataset and data processing arguments
    - **loss**: Extensible loss function framework with registry system

The training framework supports various loss functions (pointwise, pairwise, listwise)
and integrates seamlessly with the HuggingFace ecosystem.

Examples:
    Basic training setup::

        from rankers.train import RankerTrainer, RankerTrainingArguments
        from rankers.modelling import Dot
        from rankers.datasets import TrainingDataset

        args = RankerTrainingArguments(
            output_dir="./output",
            num_train_epochs=3,
            per_device_train_batch_size=32,
        )

        model = Dot.from_pretrained("bert-base-uncased")
        dataset = TrainingDataset.from_json("train.json")

        trainer = RankerTrainer(model=model, args=args, train_dataset=dataset)
        trainer.train()
"""

from . import (
    data_arguments,
    loss,
    model_arguments,
    trainer,
    training_arguments,
)
