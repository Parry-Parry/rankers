# Testing Guide for Rankers

Complete guide to the rankers test suite, including structure, usage, and best practices.

## Quick Start

```bash
# Run all tests
pytest tests/

# Run only unit tests
pytest tests/unit/

# Run only integration tests
pytest tests/integration/

# Run with coverage
pytest tests/ --cov=rankers --cov-report=html

# Run specific test
pytest tests/unit/test_trainer_init.py::TestRankerTrainerInit::test_init_with_loss_string
```

## Test Suite Overview

### 56 Total Tests Across 2 Categories

#### Unit Tests (16 tests)
Focus on individual components in isolation with mocked dependencies:
- **test_trainer_init.py** (8 tests): RankerTrainer initialization, loss functions, regularization
- **test_trainer_config.py** (8 tests): Training arguments validation and configuration
- **test_trainer_metrics.py** (includes mock metric tests): Metric computation

#### Integration Tests (40 tests)
Verify interactions between components:
- **test_trainer_evaluation.py** (5 tests): Full evaluation loops, IR metrics, prefix handling
- **test_trainer_checkpointing.py** (12 tests): Best model saving, checkpoint strategies, metric selection
- **test_evaluation_dataset.py** (13 tests): EvaluationDataset from JSONL/TREC/qrels

## Test Directory Structure

```
rankers/
├── tests/
│   ├── __init__.py
│   ├── conftest.py                  # Pytest configuration and fixtures
│   ├── README.md                    # Detailed testing documentation
│   ├── unit/
│   │   ├── __init__.py
│   │   ├── test_trainer_init.py     # Initialization tests
│   │   ├── test_trainer_config.py   # Configuration tests
│   │   └── test_trainer_metrics.py  # Metric tests
│   ├── integration/
│   │   ├── __init__.py
│   │   ├── test_trainer_evaluation.py    # Evaluation loop tests
│   │   ├── test_trainer_checkpointing.py # Best model saving tests
│   │   └── test_evaluation_dataset.py    # Dataset loader tests
│   └── fixtures/
│       ├── __init__.py
│       ├── data.py                  # Synthetic data generators
│       └── models.py                # Mock models
├── pytest.ini                       # Pytest configuration
└── TESTING.md                       # This file
```

## Key Test Areas

### 1. Trainer Initialization (8 tests)
Verifies that RankerTrainer correctly initializes with:
- String-based loss functions (via registry)
- Callable loss functions
- Invalid configurations (proper error handling)
- Regularization setup
- Model configuration updates

```python
def test_init_with_loss_string():
    args = RankerTrainingArguments(output_dir=tmpdir)
    trainer = RankerTrainer(model=model, args=args, loss_fn="margeMSE")
    assert trainer.loss is not None
```

### 2. Configuration Management (8 tests)
Validates RankerTrainingArguments handles:
- Group size configuration
- Evaluation strategies (steps/epoch)
- Save strategies
- Metric selection for best model
- Precision settings (fp16, bf16)
- Regularization parameters

```python
def test_custom_group_size():
    args = RankerTrainingArguments(output_dir=tmpdir, group_size=8)
    assert args.group_size == 8
```

### 3. Evaluation Loop (5 tests)
Tests evaluation functionality:
- Evaluation loop execution
- Custom IR metrics (nDCG, MRR, MAP)
- Metric prefix handling (`eval_` prefix)
- Multiple query evaluation

```python
def test_evaluation_loop_execution():
    output = trainer.evaluation_loop(dataset, "Test Evaluation")
    assert hasattr(output, "metrics")
```

### 4. Best Model Checkpointing (12 tests)
Validates model saving and selection:
- Best model saving with `use_best_model=True`
- Metric-based model selection
- Checkpoint frequency (steps vs epoch)
- `greater_is_better` comparison logic
- Checkpoint directory management

```python
def test_best_model_saving_with_use_best_model():
    args = RankerTrainingArguments(
        output_dir=tmpdir,
        metric_for_best_model="eval_nDCG@10",
        load_best_model_at_end=True,
    )
    trainer = RankerTrainer(model=model, args=args)
    assert args.load_best_model_at_end is True
```

### 5. EvaluationDataset Loaders (13 tests)
Tests dataset loading from multiple sources:

#### from_jsonl() (5 tests)
- JSONL training file → pseudo-qrels generation
- Custom relevance labels
- Include/exclude negatives
- Deduplication options

```python
def test_from_jsonl_creates_dataset():
    dataset = EvaluationDataset.from_jsonl(jsonl_file, corpus)
    assert hasattr(dataset, "qrels")
    assert "relevance" in dataset.qrels.columns
```

#### from_trec() (2 tests)
- TREC run file loading
- Ranking preservation

#### from_qrels() (3 tests)
- Direct qrels DataFrame loading
- Column name conversion
- Relevance value preservation

#### Common (3 tests)
- Dataset length
- Lazy text loading
- Custom key mapping

## Fixtures

All fixtures are defined in `tests/conftest.py` and `tests/fixtures/`:

### Data Fixtures
```python
@pytest.fixture
def synthetic_jsonl():
    """Synthetic JSONL training file (10 queries)"""

@pytest.fixture
def synthetic_trec():
    """Synthetic TREC format file (10 queries)"""

@pytest.fixture
def synthetic_corpus():
    """Synthetic corpus dict (50 docs, 10 queries)"""

@pytest.fixture
def synthetic_qrels():
    """Synthetic qrels DataFrame (10 queries)"""
```

### Model Fixtures
```python
@pytest.fixture
def simple_model():
    """SimpleRankerModel for testing"""

@pytest.fixture
def mock_eval_dataset():
    """Mock evaluation dataset with data and qrels"""
```

### Utility Fixtures
```python
@pytest.fixture
def temp_dir():
    """Auto-cleanup temporary directory"""
```

## Markers

Tests are automatically marked by location:

```bash
# Run only unit tests
pytest tests/ -m unit

# Run only integration tests
pytest tests/ -m integration

# Run all except slow tests
pytest tests/ -m "not slow"
```

## Coverage

Generate coverage report:

```bash
# Terminal report
pytest tests/ --cov=rankers --cov-report=term-missing

# HTML report
pytest tests/ --cov=rankers --cov-report=html
# View in: htmlcov/index.html
```

Current coverage goals:
- Unit tests: >80%
- Integration tests: Core functionality
- Overall: >70%

## Writing New Tests

### 1. Create Test File
Place in appropriate directory:
```python
# tests/unit/test_new_feature.py
import pytest
from unittest.mock import Mock, patch

class TestNewFeature:
    def test_basic_functionality(self, temp_dir):
        """Test description."""
        # Arrange

        # Act

        # Assert
```

### 2. Use Descriptive Names
```python
# Good
def test_from_jsonl_with_custom_relevance_label():

# Bad
def test_jsonl():
```

### 3. Use Fixtures
```python
def test_with_fixtures(synthetic_jsonl, simple_model, temp_dir):
    jsonl_file, records = synthetic_jsonl
    model = simple_model
    # ... test code
```

### 4. Mock External Dependencies
```python
def test_with_mocks():
    with patch("rankers.module.external_call") as mock:
        mock.return_value = expected_value
        # ... test code
```

### 5. Clean Up Resources
```python
def test_with_cleanup():
    file_path = create_temp_file()
    try:
        # ... test code
    finally:
        cleanup_temp_files([file_path])
```

## Common Testing Patterns

### Testing Error Handling
```python
def test_invalid_loss_raises_error():
    with pytest.raises(ValueError, match="Unknown loss"):
        trainer = RankerTrainer(model=model, args=args, loss_fn="invalid")
```

### Testing with Multiple Parameters
```python
@pytest.mark.parametrize("group_size,expected", [
    (2, 2),
    (4, 4),
    (8, 8),
])
def test_group_sizes(group_size, expected):
    args = RankerTrainingArguments(output_dir=tmpdir, group_size=group_size)
    assert args.group_size == expected
```

### Testing with Mocked Calls
```python
def test_with_mock_calls(mock_eval_dataset):
    # Access mock data
    assert len(mock_eval_dataset.data) == 4

    # Make assertions on mock
    mock_eval_dataset.qrels.assert_called()
```

## CI/CD Integration

### GitHub Actions Example
```yaml
name: Tests
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.10'
      - run: pip install -e ".[dev]"
      - run: pytest tests/ --cov=rankers --cov-report=xml
      - uses: codecov/codecov-action@v2
```

## Troubleshooting

### Import Errors
```bash
# Install package in editable mode
pip install -e .
```

### Temporary File Issues
Always use try/finally or context managers:
```python
try:
    jsonl_file, _ = create_synthetic_jsonl()
    # test
finally:
    cleanup_temp_files([jsonl_file])
```

### Mock Issues
Ensure you're patching the right location:
```python
# Patch where it's used, not where it's defined
with patch("rankers.train.trainer.evaluator"):
    # not: patch("ir_measures.evaluator")
```

### Assertion Failures
Use clear assertion messages:
```python
assert len(dataset.qrels) > 0, "qrels should not be empty"
```

## Performance Considerations

Tests are optimized for speed:
- Unit tests: Use mocks to avoid heavy I/O
- Integration tests: Use minimal synthetic data
- Slow tests: Marked with `@pytest.mark.slow`

Typical test run times:
- Unit tests: <5 seconds
- Integration tests: <30 seconds
- Full suite: <60 seconds

## Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [unittest.mock Documentation](https://docs.python.org/3/library/unittest.mock.html)
- [Testing Best Practices](https://docs.pytest.org/en/7.1.x/goodpractices.html)
