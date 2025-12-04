# Rankers

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green)](./LICENSE)
[![Transformers](https://img.shields.io/badge/Transformers-Compatible-orange)](https://huggingface.co/transformers/)
[![PyTerrier](https://img.shields.io/badge/PyTerrier-Compatible-orange)](https://github.com/terrier-org/pyterrier)

A lightweight, flexible library for training neural retrievers with HuggingFace `transformers` — featuring multiple ranking architectures, integrated PyTerrier support, and production-ready evaluation pipelines.

---

## 📘 Overview

**Rankers** provides a unified interface for training, evaluating, and deploying neural ranking models. Built on top of HuggingFace `transformers`, it supports multiple architectures (bi-encoders, cross-encoders, sparse retrievers, and more) while maintaining compatibility with the `transformers.Trainer` API.

- **Multiple Architectures**: Bi-encoders (Dot), Cross-encoders (CAT), Sequence-to-Sequence, Sparse models, BGE, and more
- **HuggingFace Compatible**: Drop-in `RankerTrainer` that extends `transformers.Trainer`
- **PyTerrier Integration**: Convert trained models to PyTerrier pipelines instantly
- **Production Ready**: Built-in evaluation, checkpointing, and loss functions optimized for ranking

---

## 🚀 Getting Started

### 1. Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/Parry-Parry/rankers.git
cd rankers
pip install -e .
```

Or install the latest from PyPI:

```bash
pip install rankers
```

### 2. Quick Start: Train a Bi-Encoder

```python
from rankers import RankerTrainer, RankerTrainingArguments
from rankers.modelling import Dot
from rankers.datasets import TrainingDataset, Corpus

# Load pre-trained model
model = Dot.from_pretrained("bert-base-uncased")

# Prepare datasets
corpus = Corpus.from_ir_datasets("msmarco-passage")
train_dataset = TrainingDataset("train.jsonl", corpus=corpus, group_size=4)
eval_dataset = EvaluationDataset.from_qrels("qrels.txt", corpus=corpus)

# Configure training
args = RankerTrainingArguments(
    output_dir="./output",
    num_train_epochs=3,
    per_device_train_batch_size=32,
    eval_strategy="epoch",
    save_strategy="best",
)

# Train
trainer = RankerTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    loss_fn="margin_mse",
)
trainer.train()

# Convert to PyTerrier
ranker = model.to_pyterrier()
results = ranker.transform(test_queries)
```

---

## 🧩 Supported Models

Rankers includes implementations of popular ranking architectures:

| Architecture | Class | Type | Use Case |
|--------------|-------|------|----------|
| **Dot** | `Dot` | Bi-encoder | Fast dense retrieval with separable encoders |
| **CAT** | `CAT` | Cross-encoder | High-precision ranking with joint encoding |
| **Seq2Seq** | `Seq2Seq` | Generative | Ranking via generation |
| **Sparse** | `Sparse` | Sparse Retrieval | Term-based retrieval with neural weights |
| **BGE** | `BGE` | Bi-encoder | HuggingFace BGE models |

Each model supports:
- **Training** with `RankerTrainer`
- **Evaluation** with IR metrics (nDCG, MRR, MAP, etc.)
- **PyTerrier conversion** for deployment
- **Checkpointing** with `save_pretrained()` / `from_pretrained()`

---

## 📊 Data Format

### Training Data (JSONL)

```jsonl
{"query_id": "q1", "doc_id_a": "d1", "doc_id_b": "d2"}
{"query_id": "q2", "doc_id_a": "d3", "doc_id_b": "d4"}
```

### Evaluation Data (Qrels)

```
q1 0 d1 2
q1 0 d2 1
q2 0 d3 2
```

---

## ⚙️ Training Configuration

Customize your training with `RankerTrainingArguments`:

```python
from rankers import RankerTrainingArguments

args = RankerTrainingArguments(
    output_dir="./checkpoints",
    num_train_epochs=3,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    learning_rate=2e-5,
    warmup_steps=500,

    # Evaluation & Checkpointing
    eval_strategy="steps",
    eval_steps=500,
    save_strategy="best",
    metric_for_best_model="eval_nDCG@10",
    load_best_model_at_end=True,

    # Loss & Regularization
    loss_fn="margin_mse",  # or "contrastive", "listwise", etc.
    regularizer="l2",
    regularization_weight=0.001,

    # Tracking
    report_to=["wandb"],
)
```

---

## 🎯 Loss Functions

Rankers includes multiple loss functions optimized for ranking:

- **Pairwise**: `margin_mse`, `contrastive`
- **Listwise**: `listnet`, `ndcg_loss`
- **Ranking-specific**: `triplet`, `in_batch_negatives`

---

## 🔄 Evaluation Pipeline

Built-in evaluation with IR metrics:

```python
# During training (automatic)
trainer = RankerTrainer(
    ...,
    eval_dataset=eval_dataset,
)
results = trainer.evaluate()
# Returns: {"eval_nDCG@10": 0.45, "eval_MRR": 0.52, ...}

# After training
predictions = trainer.predict(test_dataset)
# Returns: PredictionOutput with scores and metrics
```

Supported metrics (via `ir_measures`):
- nDCG@k, MRR, MAP, Recall@k, Precision@k, and more

---

## 🚀 Deployment: PyTerrier Integration

Once trained, convert to PyTerrier for deployment:

```python
# Load trained model
model = Dot.from_pretrained("./checkpoints/best_model")

# Convert to PyTerrier
ranker = model.to_pyterrier(batch_size=128)

# Use in IR pipelines
pipeline = retriever >> ranker >> reranker

# Run end-to-end ranking
results = pipeline.transform(queries_df)
```

---

## 📖 Examples

Check the `examples/` directory for complete scripts:

- **`train.bert.cat.py`**: Train a BERT-based cross-encoder
- **`train.biencoder.py`**: Train a bi-encoder with in-batch negatives
- **`train.sparse.py`**: Train a sparse neural retriever
- **`eval_and_rank.py`**: Evaluate and generate rankings

Run an example:

```bash
python examples/train.bert.cat.py \
    --model_name_or_path bert-base-uncased \
    --training_data path/to/train.jsonl \
    --output_dir ./my_ranker
```

---

## 📚 Documentation

Full API documentation is available and can be built using Sphinx:

```bash
# Install documentation dependencies
pip install -e ".[docs]"

# Build documentation
cd docs && make html

# View in browser
open _build/html/index.html
```

The documentation includes:
- Complete API reference for all modules
- Architecture deep-dives and hyperparameter guides
- Training tutorials and best practices
- PyTerrier integration examples

---

## 🧪 Testing

Comprehensive test suite with integration tests:

```bash
# Run all tests
pytest

# Run only integration tests
pytest tests/integration/

# Run with coverage
pytest --cov=rankers
```

All tests pass ✅ including:
- **89 integration & unit tests** covering full training pipeline
- **Gradient flow** verification across evaluation passes
- **Model checkpointing** and loading functionality
- **Loss computation** and backward propagation

---

## 🤝 Contributing

We welcome contributions! Please read our [contributing guidelines](CONTRIBUTING.md) for more details.

---

## 👥 Authors

- [Andrew Parry](https://github.com/Parry-Parry)

---

## 📄 License

This project is licensed under the **Apache 2.0 License** — see the [LICENSE](LICENSE) file for details.

---

## 🔗 Related Projects

- [PyTerrier](https://github.com/terrier-org/pyterrier) — Information Retrieval research platform
- [HuggingFace Transformers](https://huggingface.co/transformers/) — State-of-the-art NLP models
- [ir_measures](https://github.com/terrierteam/ir_measures) — Standard IR evaluation metrics
