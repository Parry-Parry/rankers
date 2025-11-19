# Rankers Test Suite

Comprehensive unit and integration tests for the rankers neural ranking framework.

## Structure

```
tests/
├── __init__.py
├── README.md
├── unit/
│   ├── __init__.py
│   ├── test_trainer_init.py          # RankerTrainer initialization
│   ├── test_trainer_metrics.py       # Metric computation
│   └── test_trainer_config.py        # Configuration handling
├── integration/
│   ├── __init__.py
│   ├── test_trainer_evaluation.py    # Evaluation loops
│   ├── test_trainer_checkpointing.py # Best model saving
│   └── test_evaluation_dataset.py    # EvaluationDataset loaders
└── fixtures/
    ├── __init__.py
    ├── data.py                       # Synthetic data generators
    └── models.py                     # Mock model fixtures
```

## Unit Tests

Unit tests focus on individual components in isolation:

### test_trainer_init.py
- Trainer initialization with string/callable loss functions
- Regularization setup
- Invalid configuration handling
- Model config updates

### test_trainer_metrics.py
- Metric computation from result frames
- Column renaming (qid → query_id)
- Custom metrics support
- Metric key conversion

### test_trainer_config.py
- Training arguments validation
- Group size configuration
- Evaluation strategy settings
- Precision configuration (fp16, bf16)

## Integration Tests

Integration tests verify interactions between components:

### test_trainer_evaluation.py
- Full evaluation loop execution
- Custom IR metrics (nDCG, MRR, MAP)
- Metric prefix handling
- Multi-query evaluation

### test_trainer_checkpointing.py
- Best model saving with `use_best_model=True`
- Checkpoint frequency configuration
- Epoch vs step-based checkpointing
- Metric comparison (greater_is_better)

### test_evaluation_dataset.py
- **from_jsonl()**: JSONL training file → pseudo-qrels
  - Custom relevance labels
  - Include/exclude negatives
  - Deduplication
- **from_trec()**: TREC run file loading
- **from_qrels()**: Direct qrels DataFrame
- Lazy text loading

## Test Fixtures

### data.py

Synthetic data generators for testing:

```python
from tests.fixtures.data import (
    create_synthetic_jsonl,      # Generate JSONL training file
    create_synthetic_trec,       # Generate TREC run file
    create_synthetic_corpus,     # Generate corpus dict
    create_synthetic_qrels,      # Generate qrels DataFrame
    cleanup_temp_files,          # Clean up temp files
)

# Example
jsonl_file, records = create_synthetic_jsonl(num_queries=10)
corpus = create_synthetic_corpus(num_docs=50, num_queries=10)
qrels = create_synthetic_qrels(num_queries=10)
```

### models.py

Mock models for testing:

```python
from tests.fixtures.models import SimpleRankerModel, MockPyTerrierTransformer

model = SimpleRankerModel(hidden_size=128)
pt_transformer = model.to_pyterrier(batch_size=32)
```

## Running Tests

### Run all tests
```bash
pytest tests/
```

### Run only unit tests
```bash
pytest tests/unit/
```

### Run only integration tests
```bash
pytest tests/integration/
```

### Run specific test
```bash
pytest tests/unit/test_trainer_init.py::TestRankerTrainerInit::test_init_with_loss_string
```

### Run with coverage
```bash
pytest tests/ --cov=rankers --cov-report=html
```

### Run with verbose output
```bash
pytest tests/ -v
```

### Run tests matching pattern
```bash
pytest tests/ -k "checkpoint"
```

## Test Markers

Mark tests for selective running:

```python
@pytest.mark.unit
def test_something(): ...

@pytest.mark.integration
def test_something_else(): ...

@pytest.mark.slow
def test_long_running(): ...
```

Run by marker:
```bash
pytest tests/ -m unit
pytest tests/ -m integration
pytest tests/ -m "not slow"
```

## Adding New Tests

1. Create test file in appropriate directory:
   - `tests/unit/test_*.py` for unit tests
   - `tests/integration/test_*.py` for integration tests

2. Use descriptive class and method names:
   ```python
   class TestComponentName:
       def test_specific_behavior(self): ...
       def test_edge_case(self): ...
   ```

3. Use fixtures for common setup:
   ```python
   from tests.fixtures.data import create_synthetic_jsonl

   def test_something():
       jsonl_file, records = create_synthetic_jsonl()
       # ... test code ...
   ```

4. Mock external dependencies:
   ```python
   from unittest.mock import Mock, patch

   def test_with_mocks():
       with patch("rankers.module.Class"):
           # ... test code ...
   ```

## Best Practices

1. **Isolation**: Unit tests should mock external dependencies
2. **Clarity**: Use descriptive test names that explain what's being tested
3. **Fixtures**: Reuse fixture functions for consistent test data
4. **Cleanup**: Always clean up temporary files created during tests
5. **Assertions**: Use specific assertions with helpful messages
6. **Documentation**: Add docstrings to test classes and complex test functions

## Common Issues

### Import errors
Ensure the package is installed in editable mode:
```bash
pip install -e .
```

### Temporary file issues
Always use `cleanup_temp_files()` to clean up:
```python
try:
    jsonl_file, _ = create_synthetic_jsonl()
    # ... test code ...
finally:
    cleanup_temp_files([jsonl_file])
```

### Mock issues
Use `patch()` context manager for temporary patches:
```python
with patch("module.Class") as mock_class:
    # test code
    # mock is automatically restored after
```

## CI/CD Integration

Tests are designed to work with CI/CD pipelines:

```yaml
# Example GitHub Actions
- name: Run tests
  run: pytest tests/ --cov=rankers
```

## Coverage Goals

- Unit tests: >80% coverage
- Integration tests: Core functionality coverage
- Overall target: >70% codebase coverage

Current coverage can be checked with:
```bash
pytest tests/ --cov=rankers --cov-report=term-missing
```
