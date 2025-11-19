# Rankers Examples

This directory contains comprehensive examples demonstrating how to use the Rankers library for training, evaluating, and fine-tuning neural ranking models.

## Overview

The examples are organized into three main categories:

- **[training.py](#trainingpy)** - Core training functionality
- **[evaluation.py](#evaluationpy)** - Evaluation with IR metrics
- **[lora_adapters.py](#lora_adapterspy)** - Parameter-efficient fine-tuning with LoRA

Additionally, there are existing example scripts:
- **train.bert.dot.py** - Training a Dot-product ranker with BERT
- **train.bert.cat.py** - Training a concatenation-based ranker with BERT

## training.py

Training examples covering core model training functionality.

### Examples Included

1. **train_dot_basic_example()** - Basic Dot-product model training
   - Load pretrained model with custom config
   - Create training dataset from JSONL
   - Train with margin MSE loss
   - Save trained model

2. **train_with_custom_loss_example()** - Training with specific loss functions
   - Explicitly set loss function (margin_mse, triplet loss, etc.)
   - Programmatic configuration
   - Different loss strategies

3. **train_cat_example()** - Concatenation-based ranker training
   - Cat model for query-document concatenation
   - Cat-specific data collator
   - Training pipeline for concatenation models

4. **train_with_optimization_example()** - Advanced optimization techniques
   - Gradient accumulation for larger effective batch sizes
   - Mixed precision training (fp16)
   - Custom learning rates and warmup
   - Weight decay regularization

5. **train_with_regularization_example()** - Training with regularization
   - Custom group size for triplet sampling
   - FLOPs regularization for model efficiency
   - Fine-tuning with constraints

6. **train_with_checkpointing_example()** - Checkpoint management
   - Save checkpoints at intervals
   - Load best model based on metrics
   - Training recovery from checkpoints
   - Checkpoint cleanup and management

7. **train_distributed_example()** - Multi-GPU training
   - Distributed Data Parallel (DDP) setup
   - Multi-GPU training configuration
   - Launch with torch.distributed

### Usage

Run a specific training example:

```bash
python -m examples.training
```

Or import and use in your code:

```python
from examples.training import train_dot_basic_example
train_dot_basic_example()
```

## evaluation.py

Evaluation examples demonstrating IR metric computation and evaluation strategies.

### Examples Included

1. **evaluate_with_ir_metrics_example()** - IR metrics evaluation
   - Use ir_measures for nDCG, MRR, MAP, Bpref
   - Evaluate at multiple cutoff levels
   - Comprehensive metric computation
   - Integration with trainer

2. **evaluate_from_jsonl_example()** - Evaluation from JSONL
   - Load test data from JSONL format
   - Auto-generate qrels from positive examples
   - Custom relevance label mapping
   - Negative example handling

3. **evaluate_from_qrels_example()** - Evaluation from qrels
   - Load qrels as Pandas DataFrame
   - TREC qrels format support
   - Structured relevance judgments
   - Corpus integration

4. **custom_evaluation_example()** - Custom evaluation pipeline
   - Load trained models
   - Run ranking/retrieval
   - Compute custom metrics
   - Batch predictions

5. **evaluate_checkpoints_example()** - Multi-checkpoint evaluation
   - Evaluate models at different training stages
   - Compare performance across checkpoints
   - Model selection based on metrics
   - Checkpoint comparison

6. **batch_evaluation_example()** - Efficient batch evaluation
   - Batch prediction for large test sets
   - Memory-efficient evaluation
   - Processing optimization
   - Large-scale evaluation

7. **zero_shot_evaluation_example()** - Zero-shot performance
   - Evaluate pretrained models without fine-tuning
   - Baseline performance establishment
   - Transfer learning assessment
   - Cross-domain evaluation

### Usage

Run an evaluation example:

```bash
python -m examples.evaluation
```

Or use in your code:

```python
from examples.evaluation import evaluate_with_ir_metrics_example
evaluate_with_ir_metrics_example()
```

## lora_adapters.py

LoRA (Low-Rank Adaptation) examples for parameter-efficient fine-tuning.

### Examples Included

1. **train_with_lora_basic_example()** - Basic LoRA fine-tuning
   - Initialize LoRA configuration
   - Apply LoRA to pretrained model
   - Train with adapters (memory-efficient)
   - Save adapter weights

2. **train_with_lora_custom_config_example()** - Custom LoRA configuration
   - Custom rank and alpha settings
   - Target specific attention modules
   - Fine-tune attention layers
   - Hyperparameter tuning

3. **train_lora_with_evaluation_example()** - LoRA with evaluation
   - Train with evaluation checkpoints
   - Monitor adapter performance
   - Best model selection
   - IR metric evaluation

4. **load_and_merge_lora_example()** - Load and merge adapters
   - Load pretrained LoRA adapters
   - Merge with base model
   - Inference with merged models
   - Model compression

5. **train_lora_layer_config_example()** - Layer-specific targeting
   - Target different module types
   - Fine-grained layer control
   - Custom module targeting for rankers
   - Selective adaptation

6. **multi_task_lora_example()** - Multi-task learning
   - Create adapters for multiple ranking tasks
   - Task-specific fine-tuning
   - Switch between adapters
   - Task composition

7. **train_lora_distributed_example()** - Distributed LoRA training
   - LoRA with DDP (Distributed Data Parallel)
   - Multi-GPU efficient training
   - Scaling across devices
   - Distributed checkpointing

8. **compare_lora_vs_base_example()** - Performance comparison
   - Evaluate base model
   - Evaluate LoRA-fine-tuned model
   - Compare performance metrics
   - Benchmarking

### Usage

Run a LoRA example:

```bash
python -m examples.lora_adapters
```

Or use in your code:

```python
from examples.lora_adapters import train_with_lora_basic_example
train_with_lora_basic_example()
```

### LoRA Basics

LoRA (Low-Rank Adaptation) enables efficient fine-tuning by:
- Adding trainable low-rank matrices to frozen pretrained weights
- Reducing memory usage from O(model_size) to O(rank × hidden_size)
- Reducing training time while maintaining performance
- Enabling multi-task learning with shared base model

**Benefits:**
- 10-100x fewer trainable parameters
- 2-3x faster training
- Significant memory savings
- Easy model deployment and versioning

## Quick Start

### Setting up Arguments

All examples use HuggingFace argument parsers. Create a config file or pass arguments via CLI:

```bash
python -m examples.training \
    --model_name_or_path bert-base-uncased \
    --output_dir ./output \
    --training_dataset_file train.jsonl \
    --group_size 8 \
    --num_train_epochs 3
```

### Basic Training Flow

```python
from rankers import Dot, DotConfig, DotDataCollator, TrainingDataset, RankerTrainer

# Load model
model = Dot.from_pretrained("bert-base-uncased")

# Load data
dataset = TrainingDataset("train.jsonl", group_size=8)

# Create trainer
trainer = RankerTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=DotDataCollator(model.tokenizer),
    loss_fn="margin_mse"
)

# Train
trainer.train()
```

### Basic Evaluation Flow

```python
from rankers import EvaluationDataset
from ir_measures import nDCG

# Load evaluation data
eval_dataset = EvaluationDataset.from_trec("test.trec", corpus=ir_dataset)

# Configure metrics
training_args.eval_ir_metrics = [nDCG @ 10, nDCG @ 100]

# Evaluate
metrics = trainer.evaluate(eval_dataset)
```

### Basic LoRA Flow

```python
from peft import LoraConfig, get_peft_model, TaskType

# Apply LoRA
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["query", "value"],
    task_type=TaskType.FEATURE_EXTRACTION
)
model = get_peft_model(model, lora_config)

# Train (uses less memory)
trainer.train()

# Save adapter
model.save_pretrained("./adapter")
```

## Common Arguments

### RankerTrainingArguments

Key training parameters:

- `output_dir` - Directory to save model and checkpoints
- `num_train_epochs` - Number of training epochs
- `per_device_train_batch_size` - Batch size per GPU
- `learning_rate` - Initial learning rate
- `warmup_steps` - Number of warmup steps
- `weight_decay` - Weight decay for regularization
- `group_size` - Group size for triplet sampling (1 positive + n-1 negatives)
- `loss_fn` - Loss function to use
- `eval_strategy` - Evaluation strategy (no/steps/epoch)
- `save_strategy` - Checkpoint save strategy
- `eval_ir_metrics` - IR metrics to compute (nDCG, MRR, etc.)

### RankerDataArguments

Data-related arguments:

- `training_dataset_file` - Path to training data (JSONL format)
- `test_dataset_file` - Path to test data (TREC format)
- `ir_dataset` - IR Dataset identifier for corpus (e.g., "beir/nfcorpus")
- `test_ir_dataset` - IR Dataset for test corpus
- `no_positive` - If training data has no positive examples

### RankerDotArguments / RankerCatArguments

Model-specific arguments:

- `model_name_or_path` - Pretrained model identifier
- `pooling_type` - Pooling strategy (mean, cls, etc.) for Dot models
- `model_tied` - Whether to tie query and document encoders
- `use_pooler` - Use BERT pooler layer
- `inbatch_loss` - Use in-batch negatives for contrastive loss

## Dependencies

Required packages:

```
torch
transformers
peft  # For LoRA
ir-measures  # For IR metrics
ir-datasets  # For corpus loading
datasets  # For data loading
```

Install with development dependencies:

```bash
pip install -e ".[all]"
```

## Documentation

For more information, see:
- [Rankers Documentation](https://github.com/Parry-Parry/rankers)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers)
- [PEFT Library](https://huggingface.co/docs/peft)
- [IR Measures](https://github.com/terrierteam/ir_measures)

## Contributing

Feel free to add more examples! Please ensure:
- Clear docstrings explaining the example
- Well-commented code
- Reasonable default values
- Integration with existing rankers components

## License

These examples are part of the Rankers library and follow the same license.
