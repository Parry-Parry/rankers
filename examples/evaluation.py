"""
Evaluation Examples for Rankers

This module demonstrates various evaluation techniques including:
- Evaluation with IR metrics (nDCG, MRR, MAP, etc.)
- Loading and using different dataset formats
- Custom evaluation pipelines
- Model ranking and inference
"""

import pandas as pd
from transformers import HfArgumentParser

from rankers import (
    Dot,
    DotConfig,
    DotDataCollator,
    EvaluationDataset,
    RankerDataArguments,
    RankerDotArguments,
    RankerTrainer,
    RankerTrainingArguments,
    TrainingDataset,
)


# Example 1: Evaluate with IR metrics using ir_measures
def evaluate_with_ir_metrics_example():
    """
    Evaluate model performance using IR metrics (nDCG, MRR, MAP).

    This demonstrates:
    - Loading test dataset from TREC file
    - Setting up IR metrics for evaluation
    - Computing evaluation metrics after training
    """
    from ir_measures import AP, MRR, Bpref, nDCG

    parser = HfArgumentParser((RankerDotArguments, RankerDataArguments, RankerTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Configure IR metrics for evaluation
    training_args.eval_ir_metrics = [
        nDCG @ 10,  # nDCG@10
        nDCG @ 100,  # nDCG@100
        MRR @ 10,  # Mean Reciprocal Rank@10
        AP,  # Average Precision
        Bpref,  # Binary Preference
    ]

    # Load model
    model_config = DotConfig.from_pretrained(model_args.model_name_or_path)
    model = Dot.from_pretrained(model_args.model_name_or_path, config=model_config)

    # Load training dataset
    # group_size is inferred from the dataset's first entry
    train_dataset = TrainingDataset(
        data_args.training_dataset_file,
    )

    # Load evaluation dataset from TREC format
    eval_dataset = EvaluationDataset.from_trec(
        data_args.test_dataset_file,
        corpus=data_args.test_ir_dataset,
        lazy_load_text=True,
    )

    collate_fn = DotDataCollator(model.tokenizer)

    # Create trainer with IR metrics
    trainer = RankerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collate_fn,
        loss_fn="margin_mse",
    )

    # Train the model
    trainer.train()

    # Evaluate on test set
    metrics = trainer.evaluate(eval_dataset)
    print(f"Evaluation metrics: {metrics}")


# Example 2: Evaluate from JSONL file with pseudo-qrels
def evaluate_from_jsonl_example():
    """
    Create evaluation dataset from JSONL file by generating pseudo-qrels.

    This demonstrates:
    - Loading evaluation dataset from JSONL
    - Automatically generating qrels from positive examples
    - Custom relevance label mapping
    """
    parser = HfArgumentParser((RankerDotArguments, RankerDataArguments, RankerTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    model_config = DotConfig.from_pretrained(model_args.model_name_or_path)
    model = Dot.from_pretrained(model_args.model_name_or_path, config=model_config)

    # Load evaluation dataset from JSONL
    # Automatically creates qrels from positive/negative doc IDs
    eval_dataset = EvaluationDataset.from_jsonl(
        data_args.test_dataset_file,
        query_id_key="query_id",
        positive_id_key="doc_id_a",
        negative_id_key="doc_id_b",
        relevance_label=1,  # Positive docs get relevance 1
        include_negatives=False,  # Don't include negatives in qrels
    )

    collate_fn = DotDataCollator(model.tokenizer)

    trainer = RankerTrainer(
        model=model,
        args=training_args,
        eval_dataset=eval_dataset,
        data_collator=collate_fn,
        loss_fn="margin_mse",
    )

    # Evaluate
    metrics = trainer.evaluate(eval_dataset)
    print(f"Evaluation from JSONL: {metrics}")


# Example 3: Evaluate from qrels DataFrame
def evaluate_from_qrels_example():
    """
    Evaluate using qrels provided as a Pandas DataFrame.

    This demonstrates:
    - Loading qrels as a DataFrame
    - Creating evaluation dataset from existing qrels
    - Working with structured relevance judgments
    """
    parser = HfArgumentParser((RankerDotArguments, RankerDataArguments, RankerTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    model_config = DotConfig.from_pretrained(model_args.model_name_or_path)
    model = Dot.from_pretrained(model_args.model_name_or_path, config=model_config)

    # Assume you have a qrels DataFrame
    # Format: columns [query_id, doc_id, relevance]
    qrels = pd.read_csv(
        data_args.qrels_file,
        sep="\t",
        names=["query_id", "0", "doc_id", "relevance"],
    )
    qrels = qrels[["query_id", "doc_id", "relevance"]]

    # Create evaluation dataset from qrels
    eval_dataset = EvaluationDataset.from_qrels(
        qrels,
        corpus=data_args.test_ir_dataset,
        lazy_load_text=True,
    )

    collate_fn = DotDataCollator(model.tokenizer)

    trainer = RankerTrainer(
        model=model,
        args=training_args,
        eval_dataset=eval_dataset,
        data_collator=collate_fn,
        loss_fn="margin_mse",
    )

    # Evaluate
    metrics = trainer.evaluate(eval_dataset)
    print(f"Evaluation from qrels: {metrics}")


# Example 4: Custom evaluation with ranking and ranking metrics
def custom_evaluation_example():
    """
    Custom evaluation pipeline with ranking and custom metrics.

    This demonstrates:
    - Loading trained model
    - Running ranking/retrieval
    - Computing custom metrics
    """

    parser = HfArgumentParser((RankerDotArguments, RankerDataArguments, RankerTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Load pretrained model
    model_config = DotConfig.from_pretrained(model_args.model_name_or_path)
    model = Dot.from_pretrained(
        training_args.output_dir, config=model_config
    )  # Load from training output

    # Load evaluation dataset
    eval_dataset = EvaluationDataset.from_trec(
        data_args.test_dataset_file,
        corpus=data_args.test_ir_dataset,
    )

    collate_fn = DotDataCollator(model.tokenizer)

    # Create trainer
    trainer = RankerTrainer(
        model=model,
        args=training_args,
        eval_dataset=eval_dataset,
        data_collator=collate_fn,
        loss_fn="margin_mse",
    )

    # Custom evaluation with specific metrics
    predictions = trainer.predict(eval_dataset)
    metrics = trainer.evaluate(eval_dataset)

    print(f"Predictions shape: {predictions.predictions.shape}")
    print(f"Metrics: {metrics}")


# Example 5: Evaluation at different checkpoints
def evaluate_checkpoints_example():
    """
    Evaluate model at different training checkpoints.

    This demonstrates:
    - Loading models from checkpoint directories
    - Comparing performance across training stages
    - Model selection based on evaluation metrics
    """
    from ir_measures import nDCG

    parser = HfArgumentParser((RankerDotArguments, RankerDataArguments, RankerTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Configure for checkpoint evaluation
    training_args.save_steps = 500
    training_args.eval_steps = 500
    training_args.eval_ir_metrics = [nDCG @ 10]

    model_config = DotConfig.from_pretrained(model_args.model_name_or_path)
    base_model = Dot.from_pretrained(model_args.model_name_or_path, config=model_config)

    train_dataset = TrainingDataset(data_args.training_dataset_file)
    eval_dataset = EvaluationDataset.from_trec(data_args.test_dataset_file)

    collate_fn = DotDataCollator(base_model.tokenizer)

    trainer = RankerTrainer(
        model=base_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collate_fn,
        loss_fn="margin_mse",
    )

    # Train - checkpoints saved during training
    trainer.train()

    # Load and evaluate from best checkpoint
    if training_args.load_best_model_at_end:
        best_model = Dot.from_pretrained(training_args.output_dir, config=model_config)
        print(f"Best model loaded from: {training_args.output_dir}")


# Example 6: Batch evaluation and ranking
def batch_evaluation_example():
    """
    Evaluate model in batch mode for efficiency.

    This demonstrates:
    - Batch prediction/ranking
    - Efficient evaluation of large test sets
    - Memory-efficient processing
    """
    parser = HfArgumentParser((RankerDotArguments, RankerDataArguments, RankerTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    model_config = DotConfig.from_pretrained(model_args.model_name_or_path)
    model = Dot.from_pretrained(training_args.output_dir, config=model_config)

    eval_dataset = EvaluationDataset.from_trec(
        data_args.test_dataset_file,
        corpus=data_args.test_ir_dataset,
        lazy_load_text=True,
    )

    collate_fn = DotDataCollator(model.tokenizer)

    trainer = RankerTrainer(
        model=model,
        args=training_args,
        eval_dataset=eval_dataset,
        data_collator=collate_fn,
        loss_fn="margin_mse",
    )

    # Batch prediction
    predictions = trainer.predict(eval_dataset)

    print(f"Predictions shape: {predictions.predictions.shape}")
    print(f"Predictions metrics: {predictions.metrics}")

    # Convert predictions to rankings
    # predictions.predictions typically contains shape (num_examples, 1) scores


# Example 7: Zero-shot evaluation with untrained model
def zero_shot_evaluation_example():
    """
    Evaluate a pretrained model without fine-tuning.

    This demonstrates:
    - Using pretrained models for evaluation
    - Baseline performance evaluation
    - Transfer learning assessment
    """
    from ir_measures import MRR, nDCG

    parser = HfArgumentParser((RankerDotArguments, RankerDataArguments, RankerTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Configure metrics
    training_args.eval_ir_metrics = [nDCG @ 10, MRR @ 10]

    # Load pretrained model without fine-tuning
    model_config = DotConfig.from_pretrained(
        model_args.model_name_or_path,
        pooling_type="mean",
    )
    model = Dot.from_pretrained(model_args.model_name_or_path, config=model_config)

    eval_dataset = EvaluationDataset.from_trec(
        data_args.test_dataset_file,
        corpus=data_args.test_ir_dataset,
    )

    collate_fn = DotDataCollator(model.tokenizer)

    trainer = RankerTrainer(
        model=model,
        args=training_args,
        eval_dataset=eval_dataset,
        data_collator=collate_fn,
    )

    # Evaluate without training
    metrics = trainer.evaluate(eval_dataset)
    print(f"Zero-shot evaluation: {metrics}")


if __name__ == "__main__":
    # Run basic IR metrics evaluation
    evaluate_with_ir_metrics_example()
