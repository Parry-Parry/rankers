"""
LoRA Adapter Examples for Rankers

This module demonstrates parameter-efficient fine-tuning using LoRA (Low-Rank Adaptation).
LoRA enables efficient model fine-tuning by adding low-rank trainable matrices to frozen
pretrained weights, reducing memory usage and training time.

Examples cover:
- Setting up LoRA adapters
- Fine-tuning with LoRA
- Adapter configuration and management
- Merging adapters with base models
- Multi-task learning with adapters
"""

from rankers import (
    RankerTrainingArguments,
    RankerDataArguments,
    RankerDotArguments,
    RankerTrainer,
    Dot,
    DotConfig,
    DotDataCollator,
    TrainingDataset,
    EvaluationDataset,
)
from transformers import HfArgumentParser
from peft import LoraConfig, get_peft_model, TaskType


# Example 1: Basic LoRA fine-tuning with default configuration
def train_with_lora_basic_example():
    """
    Basic LoRA fine-tuning with default settings.

    This demonstrates:
    - Initializing LoRA configuration
    - Applying LoRA to a pretrained model
    - Training with LoRA adapters
    - Memory-efficient fine-tuning
    """
    parser = HfArgumentParser(
        (RankerDotArguments, RankerDataArguments, RankerTrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Load base model
    model_config = DotConfig.from_pretrained(model_args.model_name_or_path)
    model = Dot.from_pretrained(model_args.model_name_or_path, config=model_config)

    # Set base model to eval mode before applying LoRA
    model.eval()

    # Configure LoRA
    lora_config = LoraConfig(
        r=8,  # Rank of LoRA matrices
        lora_alpha=16,  # Scaling factor
        target_modules=["query", "value"],  # Which modules to apply LoRA to
        lora_dropout=0.05,
        task_type=TaskType.FEATURE_EXTRACTION,
    )

    # Apply LoRA to model
    model = get_peft_model(model, lora_config)

    # Print trainable parameters
    model.print_trainable_parameters()

    # Load dataset
    dataset = TrainingDataset(
        data_args.training_dataset_file,
        group_size=training_args.group_size,
    )

    collate_fn = DotDataCollator(model.tokenizer)

    # Create trainer
    trainer = RankerTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collate_fn,
        loss_fn="margin_mse",
    )

    # Train with LoRA
    trainer.train()

    # Save adapter weights (not the full model)
    model.save_pretrained(training_args.output_dir)


# Example 2: LoRA with custom hyperparameters
def train_with_lora_custom_config_example():
    """
    LoRA fine-tuning with custom hyperparameters.

    This demonstrates:
    - Custom LoRA rank and alpha settings
    - Targeting specific attention modules
    - Fine-tuning attention layers only
    """
    parser = HfArgumentParser(
        (RankerDotArguments, RankerDataArguments, RankerTrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    model_config = DotConfig.from_pretrained(model_args.model_name_or_path)
    model = Dot.from_pretrained(model_args.model_name_or_path, config=model_config)

    # Configure LoRA with custom settings
    lora_config = LoraConfig(
        r=16,  # Larger rank for more expressivity
        lora_alpha=32,
        target_modules=["query", "key", "value"],  # Target more modules
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

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
    model.save_pretrained(training_args.output_dir)


# Example 3: LoRA with evaluation
def train_lora_with_evaluation_example():
    """
    LoRA fine-tuning with evaluation during training.

    This demonstrates:
    - Training with LoRA and evaluation
    - Monitoring adapter performance
    - Best model selection with adapters
    """
    from ir_measures import nDCG

    parser = HfArgumentParser(
        (RankerDotArguments, RankerDataArguments, RankerTrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Configure evaluation
    training_args.eval_strategy = "steps"
    training_args.eval_steps = 500
    training_args.save_strategy = "steps"
    training_args.save_steps = 500
    training_args.load_best_model_at_end = True
    training_args.metric_for_best_model = "loss"
    training_args.eval_ir_metrics = [nDCG @ 10]

    model_config = DotConfig.from_pretrained(model_args.model_name_or_path)
    model = Dot.from_pretrained(model_args.model_name_or_path, config=model_config)

    # Apply LoRA
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["query", "value"],
        lora_dropout=0.05,
        task_type=TaskType.FEATURE_EXTRACTION,
    )

    model = get_peft_model(model, lora_config)

    # Load datasets
    train_dataset = TrainingDataset(data_args.training_dataset_file)
    eval_dataset = EvaluationDataset.from_trec(data_args.test_dataset_file)

    collate_fn = DotDataCollator(model.tokenizer)

    trainer = RankerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collate_fn,
        loss_fn="margin_mse",
    )

    trainer.train()

    # Save best adapter
    model.save_pretrained(training_args.output_dir)


# Example 4: Load and merge LoRA adapters with base model
def load_and_merge_lora_example():
    """
    Load trained LoRA adapters and merge them with the base model.

    This demonstrates:
    - Loading pretrained models with LoRA adapters
    - Merging adapters for inference
    - Inference with merged models
    """
    from peft import PeftModel

    parser = HfArgumentParser(
        (RankerDotArguments, RankerDataArguments, RankerTrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Load base model
    model_config = DotConfig.from_pretrained(model_args.model_name_or_path)
    base_model = Dot.from_pretrained(model_args.model_name_or_path, config=model_config)

    # Load LoRA adapters
    model = PeftModel.from_pretrained(base_model, training_args.output_dir)

    print("Adapter config:")
    print(model.peft_config)

    # Merge adapters into base model
    merged_model = model.merge_and_unload()

    print(f"Model merged. Total parameters: {sum(p.numel() for p in merged_model.parameters())}")

    # Save merged model
    merged_model.save_pretrained(f"{training_args.output_dir}/merged")


# Example 5: LoRA with different layer configurations
def train_lora_layer_config_example():
    """
    LoRA fine-tuning with module-specific layer targeting.

    This demonstrates:
    - Targeting different module types
    - Fine-grained control over which layers get LoRA
    - Custom module targeting for rankers
    """
    parser = HfArgumentParser(
        (RankerDotArguments, RankerDataArguments, RankerTrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    model_config = DotConfig.from_pretrained(model_args.model_name_or_path)
    model = Dot.from_pretrained(model_args.model_name_or_path, config=model_config)

    # Configure LoRA with specific layer targets
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=[
            "query",
            "value",
            "dense",  # Also target dense layers in feedforward
        ],
        modules_to_save=["classifier"],  # Save specific modules without LoRA
        lora_dropout=0.05,
        task_type=TaskType.FEATURE_EXTRACTION,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

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
    model.save_pretrained(training_args.output_dir)


# Example 6: Multi-task learning with task-specific adapters
def multi_task_lora_example():
    """
    Multi-task learning using separate LoRA adapters for different tasks.

    This demonstrates:
    - Creating multiple adapters for different ranking tasks
    - Switching between adapters
    - Task-specific fine-tuning
    """
    from peft import PeftModel

    parser = HfArgumentParser(
        (RankerDotArguments, RankerDataArguments, RankerTrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Load base model
    model_config = DotConfig.from_pretrained(model_args.model_name_or_path)
    base_model = Dot.from_pretrained(model_args.model_name_or_path, config=model_config)

    # Create LoRA config
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["query", "value"],
        lora_dropout=0.05,
        task_type=TaskType.FEATURE_EXTRACTION,
    )

    # Task 1: Web search ranking
    model_task1 = get_peft_model(base_model, lora_config)
    dataset_task1 = TrainingDataset(data_args.training_dataset_file)

    collate_fn = DotDataCollator(model_task1.tokenizer)

    trainer_task1 = RankerTrainer(
        model=model_task1,
        args=training_args,
        train_dataset=dataset_task1,
        data_collator=collate_fn,
        loss_fn="margin_mse",
    )

    print("Training task 1 adapter...")
    trainer_task1.train()
    model_task1.save_pretrained(f"{training_args.output_dir}/task1_adapter")

    # Task 2: Can be trained similarly with different dataset
    # model_task2 = get_peft_model(base_model, lora_config)
    # ... train task2


# Example 7: LoRA with distributed training
def train_lora_distributed_example():
    """
    LoRA fine-tuning with distributed/multi-GPU training.

    This demonstrates:
    - LoRA with DDP (Distributed Data Parallel)
    - Memory-efficient multi-GPU training
    - Scaling LoRA training across devices
    """
    parser = HfArgumentParser(
        (RankerDotArguments, RankerDataArguments, RankerTrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Configure distributed training
    training_args.ddp_backend = "nccl"
    training_args.dataloader_num_workers = 4

    model_config = DotConfig.from_pretrained(model_args.model_name_or_path)
    model = Dot.from_pretrained(model_args.model_name_or_path, config=model_config)

    # Apply LoRA - works seamlessly with DDP
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["query", "value"],
        lora_dropout=0.05,
        task_type=TaskType.FEATURE_EXTRACTION,
    )

    model = get_peft_model(model, lora_config)

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
    model.save_pretrained(training_args.output_dir)


# Example 8: Compare base model vs LoRA-trained model
def compare_lora_vs_base_example():
    """
    Compare performance of base model vs LoRA fine-tuned model.

    This demonstrates:
    - Evaluating base model performance
    - Evaluating LoRA adapter performance
    - Performance comparison and benchmarking
    """
    from ir_measures import nDCG
    from peft import PeftModel

    parser = HfArgumentParser(
        (RankerDotArguments, RankerDataArguments, RankerTrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    training_args.eval_ir_metrics = [nDCG @ 10]

    # Evaluate base model
    model_config = DotConfig.from_pretrained(model_args.model_name_or_path)
    base_model = Dot.from_pretrained(model_args.model_name_or_path, config=model_config)

    eval_dataset = EvaluationDataset.from_trec(data_args.test_dataset_file)

    collate_fn = DotDataCollator(base_model.tokenizer)

    trainer_base = RankerTrainer(
        model=base_model,
        args=training_args,
        eval_dataset=eval_dataset,
        data_collator=collate_fn,
    )

    base_metrics = trainer_base.evaluate(eval_dataset)
    print(f"Base model metrics: {base_metrics}")

    # Evaluate LoRA model
    lora_model = PeftModel.from_pretrained(base_model, training_args.output_dir)

    trainer_lora = RankerTrainer(
        model=lora_model,
        args=training_args,
        eval_dataset=eval_dataset,
        data_collator=collate_fn,
    )

    lora_metrics = trainer_lora.evaluate(eval_dataset)
    print(f"LoRA model metrics: {lora_metrics}")

    # Compare
    print("\nComparison:")
    for key in base_metrics:
        base_val = base_metrics.get(key, 0)
        lora_val = lora_metrics.get(key, 0)
        if isinstance(base_val, (int, float)):
            improvement = (lora_val - base_val) / (abs(base_val) + 1e-8) * 100
            print(f"{key}: {base_val:.4f} → {lora_val:.4f} ({improvement:+.2f}%)")


if __name__ == "__main__":
    # Run basic LoRA training example
    train_with_lora_basic_example()
